"""Unit tests for the MPS module.

Coverage:
- MPS: API and invariants
- uniTensor_core: core decomposition/compression kernels
- uniTensor_utils: conversion and debug helpers

Tests are skipped automatically if `cytnx` is unavailable.
"""

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

try:
    import cytnx
except ImportError:  # pragma: no cover
    cytnx = None

THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent.parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

if cytnx is not None:
    from MPS.mps import (
        MPS,
        _check_labels,
    )
    from MPS.auto_mpo import AutoMPO
    from MPS.mps_init import random_mps
    from MPS.mps_operations import expectation, inner, mpo_sum, mps_sum
    from MPS.physical_sites import PhysicalSite
    from MPS.physical_sites.spin_half import spin_half
    from MPS.uniTensor_core import (
        assert_bond_match,
        direct_sum,
        qr_by_labels,
        scalar_from_uniTensor,
        svd_by_labels,
    )
    from MPS.uniTensor_utils import print_bond, print_bonds, to_numpy_array, to_uniTensor
    from MPS.uniTensor_utils import any_complex_tensors, is_complex_tensor
else:  # pragma: no cover
    MPS = object

    def _missing_cytnx(*_args, **_kwargs):
        raise RuntimeError("cytnx is required")

    _check_labels = _missing_cytnx
    AutoMPO = _missing_cytnx
    expectation = _missing_cytnx
    mps_sum = _missing_cytnx
    mpo_sum = _missing_cytnx
    random_mps = _missing_cytnx
    PhysicalSite = _missing_cytnx
    spin_half = _missing_cytnx
    expectation = _missing_cytnx
    assert_bond_match = _missing_cytnx
    direct_sum = _missing_cytnx
    qr_by_labels = _missing_cytnx
    scalar_from_uniTensor = _missing_cytnx
    svd_by_labels = _missing_cytnx
    print_bond = _missing_cytnx
    print_bonds = _missing_cytnx
    to_numpy_array = _missing_cytnx
    to_uniTensor = _missing_cytnx
    any_complex_tensors = _missing_cytnx
    is_complex_tensor = _missing_cytnx


def _make_site(dl: int, d: int, dr: int, start: float = 1.0) -> "cytnx.UniTensor":
    """Create a deterministic rank-3 site tensor with labels l/i/r."""
    arr = np.arange(start, start + dl * d * dr, dtype=float).reshape(dl, d, dr)
    u = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
    u.set_labels(["l", "i", "r"])
    return u


def _make_mps() -> MPS:
    """Build a small valid open-boundary 3-site MPS."""
    a0 = _make_site(1, 2, 3, start=1.0)
    a1 = _make_site(3, 2, 2, start=101.0)
    a2 = _make_site(2, 2, 1, start=201.0)
    return MPS([a0, a1, a2])


@unittest.skipIf(cytnx is None, "cytnx is required for UniTensor tests")
class TestUniTensorUtils(unittest.TestCase):
    """Tests for conversion and debug helper functions."""

    def test_to_uniTensor_and_to_numpy_array(self) -> None:
        """Round-trip conversion should preserve array values and shape."""
        arr = np.arange(24, dtype=float).reshape(2, 3, 4)
        u = to_uniTensor(arr)
        back = to_numpy_array(u)
        np.testing.assert_allclose(back, arr)

    def test_print_helpers(self) -> None:
        """Print helpers should write readable bond/tensor info."""
        tensor = _make_site(1, 2, 1)
        out = io.StringIO()
        with redirect_stdout(out):
            print_bond(tensor.bond("i"))
            print_bonds(tensor)
        text = out.getvalue()
        self.assertIn("2", text)

    def test_complex_dtype_helpers(self) -> None:
        """dtype helpers should identify real vs complex UniTensors."""
        real_t = _make_site(1, 2, 1)
        c_arr = np.array([[[1.0 + 1.0j], [0.0 + 0.0j]]], dtype=complex)
        c_t = cytnx.UniTensor(cytnx.from_numpy(c_arr), rowrank=2)
        c_t.set_labels(["l", "i", "r"])
        self.assertFalse(is_complex_tensor(real_t))
        self.assertTrue(is_complex_tensor(c_t))
        self.assertFalse(any_complex_tensors([real_t, real_t]))
        self.assertTrue(any_complex_tensors([real_t, c_t]))


@unittest.skipIf(cytnx is None, "cytnx is required for UniTensor tests")
class TestUniTensorCore(unittest.TestCase):
    """Tests for core linear-algebra and bond-matching kernels."""

    def test_scalar_from_uniTensor(self) -> None:
        """Scalar extraction should read one-element UniTensor correctly."""
        scalar = cytnx.UniTensor(cytnx.from_numpy(np.array([3.5])), rowrank=0)
        self.assertAlmostEqual(scalar_from_uniTensor(scalar), 3.5)

    def test_assert_bond_match(self) -> None:
        """Bond matcher should pass on equal bonds and fail on mismatched ones."""
        t1 = _make_site(1, 2, 3)
        t2 = _make_site(1, 2, 3)
        assert_bond_match(t1.bond("i"), t2.bond("i"))
        with self.assertRaises(ValueError):
            bad = _make_site(1, 3, 3)
            assert_bond_match(t1.bond("i"), bad.bond("i"))

    def test_svd_by_labels(self) -> None:
        """svd_by_labels should split tensor and reconstruct the original tensor."""
        arr = np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2)
        t = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
        a1, a2, dw = svd_by_labels(
            t,
            row_labels=["0", "1"],
            absorb="right",
            dim=4,
            cutoff=0.0,
            aux_label="x",
        )
        self.assertIsInstance(a1, cytnx.UniTensor)
        self.assertIsInstance(a2, cytnx.UniTensor)
        self.assertGreaterEqual(dw, 0.0)
        self.assertLessEqual(dw, 1.0)
        self.assertEqual(a1.labels(), ["0", "1", "x"])
        self.assertEqual(a2.labels(), ["x", "2", "3"])
        rebuilt = cytnx.Contract(a1, a2)
        np.testing.assert_allclose(to_numpy_array(rebuilt), arr, atol=1e-12)

    def test_svd_by_labels_absorb_left(self) -> None:
        """svd_by_labels with absorb='left' should absorb s into left tensor."""
        arr = np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2)
        t = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
        a1, a2, dw = svd_by_labels(
            t,
            row_labels=["0", "1"],
            absorb="left",
            dim=4,
            cutoff=0.0,
            aux_label="x",
        )
        self.assertIsInstance(a1, cytnx.UniTensor)
        self.assertIsInstance(a2, cytnx.UniTensor)
        self.assertGreaterEqual(dw, 0.0)
        self.assertLessEqual(dw, 1.0)
        self.assertEqual(a1.labels(), ["0", "1", "x"])
        self.assertEqual(a2.labels(), ["x", "2", "3"])
        rebuilt = cytnx.Contract(a1, a2)
        np.testing.assert_allclose(to_numpy_array(rebuilt), arr, atol=1e-12)

    def test_svd_by_labels_invalid_absorb(self) -> None:
        """svd_by_labels should raise ValueError for unknown absorb value."""
        arr = np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2)
        t = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
        with self.assertRaises(ValueError):
            svd_by_labels(t, row_labels=["0", "1"], absorb="both")

    def test_svd_by_labels_negative_cutoff_raises(self) -> None:
        """svd_by_labels should reject cutoff < 0."""
        arr = np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2)
        t = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
        with self.assertRaises(ValueError):
            svd_by_labels(t, row_labels=["0", "1"], absorb="right", cutoff=-1e-12)

    def test_svd_by_labels_zero_tensor_raises(self) -> None:
        """svd_by_labels should raise when all singular values are zero."""
        arr = np.zeros((2, 3, 2, 2), dtype=float)
        t = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
        with self.assertRaises(ValueError):
            svd_by_labels(t, row_labels=["0", "1"], absorb="right", cutoff=0.0)

    def test_split_row_col_labels_errors(self) -> None:
        """_split_row_col_labels error paths should propagate through svd_by_labels."""
        arr = np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2)
        t = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
        # neither row_labels nor col_labels given
        with self.assertRaises(ValueError):
            svd_by_labels(t, absorb="right")
        # row_labels covers every label → col group becomes empty
        with self.assertRaises(ValueError):
            svd_by_labels(t, row_labels=["0", "1", "2", "3"], absorb="right")
        # row_labels contains a label not present in the tensor → union != full label set
        with self.assertRaises(ValueError):
            svd_by_labels(t, row_labels=["0", "nosuchlabel"], absorb="right")

    def test_svd_by_labels_no_absorb(self) -> None:
        """svd_by_labels with absorb=None should return (left, s, right, discarded)."""
        arr = np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2)
        t = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
        a1, s, a2, dw = svd_by_labels(t, row_labels=["0", "1"], aux_label="x")
        self.assertIsInstance(s, cytnx.UniTensor)
        self.assertGreaterEqual(dw, 0.0)
        # left connects to s via "x"; s connects to right via "x_r"
        self.assertEqual(a1.labels()[-1], "x")
        self.assertEqual(a2.labels()[0], "x_r")
        rebuilt = cytnx.Contract(cytnx.Contract(a1, s), a2)
        np.testing.assert_allclose(to_numpy_array(rebuilt), arr, atol=1e-12)

    def test_qr_by_labels(self) -> None:
        """qr_by_labels should split tensor and reconstruct the original tensor."""
        tensor = _make_site(3, 2, 2)
        q, r = qr_by_labels(tensor, row_labels=["l", "i"], aux_label="x")
        self.assertEqual(q.labels(), ["l", "i", "x"])
        self.assertEqual(r.labels(), ["x", "r"])
        self.assertEqual(q.bond("x").dim(), r.bond("x").dim())
        rebuilt = cytnx.Contract(q, r)
        np.testing.assert_allclose(to_numpy_array(rebuilt), to_numpy_array(tensor))


@unittest.skipIf(cytnx is None, "cytnx is required for UniTensor tests")
class TestDirectSum(unittest.TestCase):
    """Tests for direct_sum of QN UniTensors."""

    def setUp(self):
        self.sym = [cytnx.Symmetry.U1()]

    def _make_rank2(self, i_degs, j_degs=None):
        """Rank-2 UniTensor with labels ["i","j"], rowrank=1."""
        sym = self.sym
        b_i = cytnx.Bond(cytnx.BD_IN,  [[0], [1]], i_degs, sym)
        b_j = cytnx.Bond(cytnx.BD_OUT, [[0], [1]], j_degs or [1, 1], sym)
        t = cytnx.UniTensor([b_i, b_j], labels=["i", "j"], rowrank=1)
        cytnx.random.uniform_(t, 0., 1.)
        return t

    def _make_rank4_mpo(self, l_degs, r_degs):
        """Rank-4 MPO-like UniTensor with labels ["l","ip","i","r"], rowrank=2."""
        sym = self.sym
        b_l  = cytnx.Bond(cytnx.BD_IN,  [[0], [1]], l_degs, sym)
        b_ip = cytnx.Bond(cytnx.BD_IN,  [[0], [1]], [1, 1], sym)
        b_i  = cytnx.Bond(cytnx.BD_OUT, [[0], [1]], [1, 1], sym)
        b_r  = cytnx.Bond(cytnx.BD_OUT, [[0], [1]], r_degs, sym)
        t = cytnx.UniTensor([b_l, b_ip, b_i, b_r], labels=["l", "ip", "i", "r"], rowrank=2)
        cytnx.random.uniform_(t, 0., 1.)
        return t

    def test_rank2_inner_product(self):
        """<A⊕B | C⊕D> == <A|C> + <B|D> for rank-2 single-bond direct sum."""
        A = self._make_rank2([2, 3])
        B = self._make_rank2([1, 2])
        C = self._make_rank2([2, 3])
        D = self._make_rank2([1, 2])

        AB = direct_sum(A, B, ["i"], ["i"], ["i"])
        CD = direct_sum(C, D, ["i"], ["i"], ["i"])

        inner_AB_CD = cytnx.Contract(AB.Dagger(), CD).item()
        inner_AC    = cytnx.Contract(A.Dagger(), C).item()
        inner_BD    = cytnx.Contract(B.Dagger(), D).item()
        self.assertAlmostEqual(inner_AB_CD, inner_AC + inner_BD, places=10)

    def test_rank4_mpo_norm(self):
        """‖A⊕B‖² == ‖A‖² + ‖B‖² for rank-4 MPO-like two-bond direct sum."""
        mA = self._make_rank4_mpo(l_degs=[2, 1], r_degs=[1, 2])
        mB = self._make_rank4_mpo(l_degs=[1, 3], r_degs=[2, 1])

        mAB = direct_sum(mA, mB, ["l", "r"], ["l", "r"], ["l", "r"])

        norm_AB_sq = mAB.Norm().item() ** 2
        norm_A_sq  = mA.Norm().item() ** 2
        norm_B_sq  = mB.Norm().item() ** 2
        self.assertAlmostEqual(norm_AB_sq, norm_A_sq + norm_B_sq, places=10)

    def test_re_label_renamed(self):
        """Output bond carries the name given in re_labels, even when it differs from sum_labels."""
        A = self._make_rank2([2, 3])
        B = self._make_rank2([1, 2])

        AB = direct_sum(A, B, ["i"], ["i"], ["new_i"])
        self.assertIn("new_i", list(AB.labels()))
        self.assertNotIn("i", list(AB.labels()))

    def test_error_length_mismatch(self):
        """Mismatched list lengths should raise ValueError."""
        A = self._make_rank2([2, 3])
        B = self._make_rank2([1, 2])
        with self.assertRaises(ValueError):
            direct_sum(A, B, ["i"], ["i"], ["i", "extra"])

    def test_error_non_sum_label_mismatch(self):
        """A and B having different non-sum labels should raise ValueError."""
        sym = self.sym
        b_i = cytnx.Bond(cytnx.BD_IN,  [[0], [1]], [2, 3], sym)
        b_jA = cytnx.Bond(cytnx.BD_OUT, [[0], [1]], [1, 1], sym)
        b_jB = cytnx.Bond(cytnx.BD_OUT, [[0], [1]], [1, 1], sym)
        A = cytnx.UniTensor([b_i, b_jA], labels=["i", "j"], rowrank=1)
        B = cytnx.UniTensor([b_i, b_jB], labels=["i", "k"], rowrank=1)  # "k" != "j"
        with self.assertRaises(ValueError):
            direct_sum(A, B, ["i"], ["i"], ["i"])

    def test_error_non_sum_bond_mismatch(self):
        """Same non-sum label but incompatible bond content should raise ValueError."""
        sym = self.sym
        b_iA = cytnx.Bond(cytnx.BD_IN,  [[0], [1]], [2, 3], sym)
        b_iB = cytnx.Bond(cytnx.BD_IN,  [[0], [1]], [1, 2], sym)
        b_j1 = cytnx.Bond(cytnx.BD_OUT, [[0], [1]], [1, 1], sym)
        b_j2 = cytnx.Bond(cytnx.BD_OUT, [[0], [1]], [2, 1], sym)  # different degs
        A = cytnx.UniTensor([b_iA, b_j1], labels=["i", "j"], rowrank=1)
        B = cytnx.UniTensor([b_iB, b_j2], labels=["i", "j"], rowrank=1)
        with self.assertRaises(ValueError):
            direct_sum(A, B, ["i"], ["i"], ["i"])

    def test_error_re_label_clashes_non_sum(self):
        """re_label equal to a non-summed label should raise ValueError."""
        A = self._make_rank2([2, 3])
        B = self._make_rank2([1, 2])
        with self.assertRaises(ValueError):
            direct_sum(A, B, ["i"], ["i"], ["j"])  # "j" is a non-sum label


@unittest.skipIf(cytnx is None, "cytnx is required for UniTensor tests")
class TestMPSSum(unittest.TestCase):
    """Tests for mps_sum."""

    def test_inner_product_factorizes(self):
        """inner(sum(α,β), sum(φ,χ)) == inner(α,φ) + inner(β,χ)."""
        site = spin_half(qn="Sz")
        alpha = site.product_state([1, 0, 1, 0])
        phi   = site.product_state([1, 0, 1, 0])
        beta  = site.product_state([0, 1, 0, 1])
        chi   = site.product_state([0, 1, 0, 1])

        sumAB = mps_sum(alpha, beta)
        sumPC = mps_sum(phi, chi)

        self.assertAlmostEqual(
            inner(sumAB, sumPC),
            inner(alpha, phi) + inner(beta, chi),
            places=10,
        )

    def test_bond_dims(self):
        """Virtual bond dims of mps_sum equal the sum of the two MPS bond dims."""
        site = spin_half(qn="Sz")
        psi = site.product_state([1, 0, 1, 0])
        phi = site.product_state([0, 1, 0, 1])
        result = mps_sum(psi, phi)
        for k in range(len(psi) - 1):
            expected = psi[k].bond("r").dim() + phi[k].bond("r").dim()
            self.assertEqual(result[k].bond("r").dim(), expected)


@unittest.skipIf(cytnx is None, "cytnx is required for UniTensor tests")
class TestMPOSum(unittest.TestCase):
    """Tests for mpo_sum."""

    def test_expectation_factorizes(self):
        """expectation(ψ, mpo_sum(H1, H2), ψ) == expectation(ψ, H1, ψ) + expectation(ψ, H2, ψ)."""
        site = spin_half(qn="Sz")
        N = 4
        J = 1.0

        ampo_ising = AutoMPO(N, site)
        ampo_xy    = AutoMPO(N, site)
        for i in range(N - 1):
            ampo_ising.add(J,       "Sz", i, "Sz", i + 1)
            ampo_xy.add(J / 2, "Sp", i, "Sm", i + 1)
            ampo_xy.add(J / 2, "Sm", i, "Sp", i + 1)
        H_ising = ampo_ising.to_mpo()
        H_xy    = ampo_xy.to_mpo()
        H_sum   = mpo_sum(H_ising, H_xy)

        psi = site.product_state([1, 0, 1, 0])

        self.assertAlmostEqual(
            expectation(psi, H_sum, psi),
            expectation(psi, H_ising, psi) + expectation(psi, H_xy, psi),
            places=10,
        )


@unittest.skipIf(cytnx is None, "cytnx is required for UniTensor tests")
class TestMPS(unittest.TestCase):
    """Tests for high-level MPS behavior and invariants."""

    def test_init_rejects_empty_list(self) -> None:
        """MPS constructor should raise ValueError for empty site list."""
        with self.assertRaises(ValueError):
            MPS([])

    def test_init_rejects_wrong_type(self) -> None:
        """MPS constructor should raise TypeError when a site is not UniTensor."""
        with self.assertRaises(TypeError):
            MPS([np.zeros((1, 2, 1))])

    def test_check_labels(self) -> None:
        """Site label validator should accept l/i/r and reject invalid labels."""
        t = _make_site(1, 2, 1)
        _check_labels(t, 0)
        bad = t.clone()
        bad.set_labels(["x", "y", "z"])
        with self.assertRaises(ValueError):
            _check_labels(bad, 0)

    def test_basic_sequence_protocol_and_repr(self) -> None:
        """MPS should expose sequence protocol and shape summary properties."""
        mps = _make_mps()
        self.assertEqual(len(mps), 3)
        self.assertEqual(sum(1 for _ in mps), 3)
        self.assertIsInstance(mps[0], cytnx.UniTensor)
        self.assertIn("phys_dim", repr(mps))
        self.assertEqual(mps.phys_dims, [2, 2, 2])
        self.assertEqual(mps.bond_dims, [1, 3, 2, 1])
        self.assertEqual(mps.max_dim, 3)
        self.assertEqual((mps.center_left, mps.center_right), (0, 2))
        self.assertIsNone(mps.center)

    def test_setitem_validation_and_rollback(self) -> None:
        """Invalid site replacement should raise and rollback previous tensor."""
        mps = _make_mps()
        original = mps[1].clone()
        with self.assertRaises(ValueError):
            mps[1] = _make_site(4, 2, 2)  # left bond mismatches site 0 right bond=3
        self.assertEqual(mps[1].bond("l").dim(), original.bond("l").dim())

    def test_setitem_updates_center_window(self) -> None:
        """Manual replacement should expand center window to include changed site."""
        mps = _make_mps().orthogonalize()  # center at site 0
        self.assertEqual((mps.center_left, mps.center_right), (0, 0))
        mps[1] = _make_site(3, 2, 2, start=500.0)
        self.assertEqual((mps.center_left, mps.center_right), (0, 1))
        self.assertIsNone(mps.center)

    def test_copy_inner_norm_normalize(self) -> None:
        """copy/inner/norm/normalize should behave consistently."""
        mps = _make_mps()
        mps_copy = mps.copy()
        self.assertIsNot(mps_copy[0], mps[0])
        val = inner(mps, mps_copy)
        self.assertTrue(np.isfinite(float(np.real(val))))
        n0 = mps.norm()
        self.assertGreater(n0, 0.0)
        with self.assertRaises(ValueError):
            mps.normalize()  # no single center yet
        mps.orthogonalize()
        mps.normalize()
        self.assertAlmostEqual(mps.norm(), 1.0, places=8)

    def test_normalize_requires_single_center(self) -> None:
        """normalize should raise when center window spans multiple sites."""
        mps = _make_mps()
        self.assertIsNone(mps.center)
        with self.assertRaises(ValueError):
            mps.normalize()

    def test_move_center_and_orthogonalize(self) -> None:
        """Center-moving and orthogonalization should place center as requested."""
        mps = _make_mps()
        mps.move_center(2)
        self.assertEqual((mps.center_left, mps.center_right), (2, 2))
        self.assertEqual(mps.center, 2)
        mps.move_center(1)
        self.assertEqual((mps.center_left, mps.center_right), (1, 1))
        self.assertEqual(mps.center, 1)
        mps.orthogonalize()
        self.assertEqual((mps.center_left, mps.center_right), (1, 1))
        self.assertEqual(mps.center, 1)
        mps.orthogonalize(center=2)
        self.assertEqual((mps.center_left, mps.center_right), (2, 2))
        self.assertEqual(mps.center, 2)
        with self.assertRaises(IndexError):
            mps.orthogonalize(center=99)

    def test_move_center_from_no_center(self) -> None:
        """move_center should auto-orthogonalize when center window spans multiple sites."""
        mps = _make_mps()
        self.assertIsNone(mps.center)
        mps.move_center(2)
        self.assertEqual(mps.center, 2)

    def test_move_center_out_of_range(self) -> None:
        """move_center should raise IndexError for out-of-range site."""
        mps = _make_mps()
        with self.assertRaises(IndexError):
            mps.move_center(-1)
        with self.assertRaises(IndexError):
            mps.move_center(len(mps))

    def test_check_left_right_orthonormal(self) -> None:
        """Orthonormal checker should raise when outside-window constraints fail."""
        mps = _make_mps().orthogonalize()
        # Should pass without raising for a canonicalized state.
        mps.check_left_right_orthonormal()

        # Break right-orthonormality on a site outside the center window.
        broken = _make_mps().orthogonalize().move_center(1)
        broken.tensors[-1] = _make_site(2, 2, 1, start=999.0)
        with self.assertRaises(ValueError):
            broken.check_left_right_orthonormal()

    def test_check_compatible_uses_bond_match(self) -> None:
        """Compatibility check should fail when physical bonds do not match."""
        m1 = _make_mps()
        m2 = _make_mps()
        m1._check_compatible(m2)
        bad1 = _make_site(1, 3, 3, start=10.0)
        bad2 = _make_site(3, 2, 2, start=20.0)
        bad3 = _make_site(2, 2, 1, start=30.0)
        m3 = MPS([bad1, bad2, bad3])
        with self.assertRaises(ValueError):
            m1._check_compatible(m3)

    def test_factory_random_mps_shape(self) -> None:
        """random_mps should produce an MPS with the requested dimensions."""
        mps = random_mps(num_sites=4, phys_dim=2, bond_dim=3, seed=0)
        self.assertEqual(len(mps), 4)
        self.assertEqual(mps.phys_dims, [2, 2, 2, 2])
        # Interior bond dims capped by SVD rank; boundary dims must be 1.
        self.assertEqual(mps.bond_dims[0], 1)
        self.assertEqual(mps.bond_dims[-1], 1)

    def test_factory_random_mps_normalized(self) -> None:
        """random_mps with normalize=True should produce a unit-norm MPS with a single center."""
        mps = random_mps(num_sites=3, phys_dim=2, bond_dim=4, normalize=True, seed=1)
        self.assertIsNotNone(mps.center)
        self.assertAlmostEqual(mps.norm(), 1.0, places=8)

    def test_factory_random_mps_not_normalized(self) -> None:
        """random_mps with normalize=False should not enforce canonical form."""
        mps = random_mps(num_sites=3, phys_dim=2, bond_dim=4, normalize=False, seed=2)
        # Norm should still be finite and positive, but not necessarily 1.
        nrm = mps.norm()
        self.assertGreater(nrm, 0.0)
        self.assertTrue(np.isfinite(nrm))

    def test_is_complex_false_for_real_mps(self) -> None:
        """Real-valued MPS should report is_complex == False."""
        mps = random_mps(num_sites=3, phys_dim=2, bond_dim=3, normalize=False, seed=3)
        self.assertFalse(mps.is_complex)

    def test_is_complex_true_for_complex_mps(self) -> None:
        """Complex-valued MPS should report is_complex == True."""
        mps = random_mps(
            num_sites=3, phys_dim=2, bond_dim=3, dtype=complex, normalize=False, seed=4
        )
        self.assertTrue(mps.is_complex)

    def test_inner_complex_mps_does_not_mismatch_env_dtype(self) -> None:
        """inner on complex MPS should run without real/complex env mismatch."""
        mps = random_mps(
            num_sites=3, phys_dim=2, bond_dim=3, dtype=complex, normalize=False, seed=5
        )
        val = inner(mps, mps)
        self.assertTrue(np.isfinite(float(np.real(val))))

    def test_factory_random_mps_seed_reproducibility(self) -> None:
        """random_mps with the same seed should produce identical tensors."""
        m1 = random_mps(num_sites=3, phys_dim=2, bond_dim=4, normalize=False, seed=42)
        m2 = random_mps(num_sites=3, phys_dim=2, bond_dim=4, normalize=False, seed=42)
        for t1, t2 in zip(m1, m2):
            np.testing.assert_array_equal(to_numpy_array(t1), to_numpy_array(t2))

    def test_factory_random_mps_rejects_invalid_params(self) -> None:
        """random_mps should raise ValueError for non-positive dimension arguments."""
        with self.assertRaises(ValueError):
            random_mps(num_sites=0, phys_dim=2, bond_dim=3)
        with self.assertRaises(ValueError):
            random_mps(num_sites=3, phys_dim=0, bond_dim=3)
        with self.assertRaises(ValueError):
            random_mps(num_sites=3, phys_dim=2, bond_dim=0)

    def test_factory_product_state(self) -> None:
        """Factory product_state should create bond-1 one-hot basis tensors."""
        config = [0, 1, 2]
        phys_dim = 3
        site = PhysicalSite(cytnx.Bond(phys_dim, cytnx.BD_IN))
        mps = site.product_state(config)
        self.assertEqual(len(mps), 3)
        self.assertEqual(mps.phys_dims, [phys_dim, phys_dim, phys_dim])
        self.assertEqual(mps.bond_dims, [1, 1, 1, 1])

        for site, state in enumerate(config):
            arr = to_numpy_array(mps[site])
            expected = np.zeros(phys_dim, dtype=float)
            expected[state] = 1.0
            np.testing.assert_allclose(arr.reshape(-1), expected)

    def test_factory_product_state_rejects_invalid_config(self) -> None:
        """Factory product_state should reject invalid config entries."""
        site = PhysicalSite(cytnx.Bond(2, cytnx.BD_IN))
        with self.assertRaises(ValueError):
            site.product_state([])
        with self.assertRaises(ValueError):
            site.product_state([0, 2])
        with self.assertRaises(TypeError):
            site.product_state([0, 1.0])

    def test_factory_product_state_rejects_non_python_int(self) -> None:
        """Factory product_state should reject numpy integer objects."""
        site = PhysicalSite(cytnx.Bond(2, cytnx.BD_IN))
        with self.assertRaises(TypeError):
            site.product_state([0, np.int64(1)])

    def test_factory_product_state_qn(self) -> None:
        """QN product state should build blockform sites for a valid basis index."""
        sym = cytnx.Symmetry.Zn(2)
        phys = cytnx.Bond(cytnx.BD_IN, [[0], [1]], [1, 1], [sym])
        site = PhysicalSite(phys)
        mps = site.product_state([0, 0, 0])
        self.assertEqual(len(mps), 3)
        self.assertEqual(mps.bond_dims, [1, 1, 1, 1])
        for tensor in mps:
            self.assertTrue(tensor.is_blockform())
            self.assertEqual(tensor.Nblocks(), 1)
            block = tensor.get_block_(0)
            self.assertAlmostEqual(float(block[0, 0, 0].item()), 1.0)

    def test_factory_product_state_qn_accepts_nonzero_sector_index(self) -> None:
        """PhysicalSite.product_state supports any valid local basis index."""
        sym = cytnx.Symmetry.Zn(2)
        phys = cytnx.Bond(cytnx.BD_IN, [[0], [1]], [1, 1], [sym])
        site = PhysicalSite(phys)
        mps = site.product_state([1, 1])
        self.assertEqual(len(mps), 2)
        self.assertEqual(mps.total_qn, [0])


if __name__ == "__main__":
    unittest.main()
