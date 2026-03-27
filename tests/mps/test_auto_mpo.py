"""Tests for AutoMPO."""

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
    from tests.helpers.heisenberg import heisenberg_mpo

SKIP = unittest.skipIf(cytnx is None, "cytnx not available")


# ---------------------------------------------------------------------------
# Helper: extract full dense matrix from MPO via full contraction
# ---------------------------------------------------------------------------

def _get_dense_numpy(ut):
    """Return the full dense numpy array for a UniTensor (dense or QN)."""
    if ut.is_blockform():
        # Convert QN tensor to dense first
        shape = list(ut.shape())
        arr = np.zeros(shape, dtype=ut.get_block().numpy().dtype)
        # Iterate over all blocks and place them into the dense array
        # Build index maps: for each bond, map flat_idx -> (sector, offset)
        from MPS.uniTensor_core import _bond_sector_at
        bonds = [ut.bond(label) for label in ut.labels()]
        n = len(bonds)

        from itertools import product as iproduct
        for idx in iproduct(*[range(s) for s in shape]):
            try:
                val = ut.at(list(idx)).value
            except (RuntimeError, ValueError):
                # Invalid flat index for this QN block structure.
                continue
            arr[idx] = val
        return arr
    return ut.get_block().numpy()


def mpo_full_matrix(mpo):
    """Return the d^N × d^N dense matrix by pure numpy contraction.

    W[0]["l"] and W[-1]["r"] each have dim=1, so the boundary condition is
    simply vec = [1.0] on the left and val = vec[0] on the right.
    """
    N = len(mpo)
    d = mpo.phys_dims[0]

    # Get W arrays in order [l, ip, i, r]
    W_arrs = []
    for p in range(N):
        w_perm = mpo[p].permute(["l", "ip", "i", "r"])
        W_arrs.append(_get_dense_numpy(w_perm))   # (Dl, d, d, Dr)

    from itertools import product as iproduct

    out_dtype = np.result_type(*[arr.dtype for arr in W_arrs])
    total_dim = d**N
    mat = np.zeros((total_dim, total_dim), dtype=out_dtype)

    for bra_idx in iproduct(range(d), repeat=N):
        for ket_idx in iproduct(range(d), repeat=N):
            # W[0]["l"] dim=1 → start with scalar [1.0]; W[-1]["r"] dim=1 → read vec[0]
            vec = np.array([1.0], dtype=out_dtype)
            for p in range(N):
                M = W_arrs[p][:, bra_idx[p], ket_idx[p], :]  # (Dl, Dr)
                vec = M.T @ vec   # (Dr,)
            val = vec[0]

            bra_flat = sum(bra_idx[p] * d**(N - 1 - p) for p in range(N))
            ket_flat = sum(ket_idx[p] * d**(N - 1 - p) for p in range(N))
            mat[bra_flat, ket_flat] = val

    return mat


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

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

    def test_sites_must_be_increasing(self):
        with self.assertRaises(ValueError):
            self.ampo.add(1.0, "Sz", 1, "Sz", 0)

    def test_sites_must_be_strictly_increasing(self):
        with self.assertRaises(ValueError):
            self.ampo.add(1.0, "Sz", 1, "Sz", 1)

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
            ampo.add(1.0, "Sm", 1)      # charge -1 ≠ +1

    def test_no_terms_raises(self):
        with self.assertRaises(RuntimeError):
            self.ampo.to_mpo()


# ---------------------------------------------------------------------------
# Dense MPO correctness: compare with heisenberg_mpo
# ---------------------------------------------------------------------------

@SKIP
class TestAutoMPODenseHeisenberg(unittest.TestCase):

    def _heisenberg_ampo(self, N, J=1.0, delta=1.0, h=0.0):
        site = spin_half()
        ampo = AutoMPO(N, site)
        for i in range(N - 1):
            ampo.add(J * delta, "Sz", i, "Sz", i + 1)
            ampo.add(J / 2,     "Sp", i, "Sm", i + 1)
            ampo.add(J / 2,     "Sm", i, "Sp", i + 1)
        for i in range(N):
            ampo.add(h, "Sz", i)
        return ampo.to_mpo()

    def test_matrix_matches_heisenberg_mpo_N4(self):
        N = 4
        H_ref  = heisenberg_mpo(N, J=1.0, delta=1.0, h=0.0)
        H_auto = self._heisenberg_ampo(N, J=1.0, delta=1.0, h=0.0)
        ref_mat  = mpo_full_matrix(H_ref)
        auto_mat = mpo_full_matrix(H_auto)
        np.testing.assert_allclose(auto_mat, ref_mat, atol=1e-12,
                                   err_msg="Dense AutoMPO ≠ heisenberg_mpo")

    def test_matrix_matches_heisenberg_mpo_anisotropic(self):
        N = 4
        J, delta, h = 1.5, 0.7, 0.3
        H_ref  = heisenberg_mpo(N, J=J, delta=delta, h=h)
        H_auto = self._heisenberg_ampo(N, J=J, delta=delta, h=h)
        ref_mat  = mpo_full_matrix(H_ref)
        auto_mat = mpo_full_matrix(H_auto)
        np.testing.assert_allclose(auto_mat, ref_mat, atol=1e-12,
                                   err_msg="Dense AutoMPO ≠ heisenberg_mpo (anisotropic)")

    def test_mpo_structure(self):
        N = 4
        H = self._heisenberg_ampo(N)
        self.assertEqual(len(H), N)
        for p in range(N):
            self.assertEqual(set(H[p].labels()), {"l", "ip", "i", "r"})

    def test_single_site_field(self):
        """h*Sz on each site."""
        N = 3
        site = spin_half()
        ampo = AutoMPO(N, site)
        for i in range(N):
            ampo.add(2.0, "Sz", i)
        H = ampo.to_mpo()
        # Full matrix should be 2 * sum_i Sz_i
        mat = mpo_full_matrix(H)
        # Build reference: 2*(Sz⊗I⊗I + I⊗Sz⊗I + I⊗I⊗Sz)
        Sz = np.diag([0.5, -0.5])
        I2 = np.eye(2)
        ref = 2.0 * (
            np.kron(np.kron(Sz, I2), I2) +
            np.kron(np.kron(I2, Sz), I2) +
            np.kron(np.kron(I2, I2), Sz)
        )
        np.testing.assert_allclose(mat, ref, atol=1e-12)

    def test_next_nearest_neighbor(self):
        """J2 * Sz_i Sz_{i+2} (non-adjacent)."""
        N = 4
        J2 = 0.5
        site = spin_half()
        ampo = AutoMPO(N, site)
        for i in range(N - 2):
            ampo.add(J2, "Sz", i, "Sz", i + 2)
        H = ampo.to_mpo()
        mat = mpo_full_matrix(H)
        # Build reference
        Sz = np.diag([0.5, -0.5])
        I2 = np.eye(2)
        ops = [Sz, I2, I2, I2]

        def kron4(a, b, c, dd):
            return np.kron(np.kron(np.kron(a, b), c), dd)

        ref = J2 * (
            kron4(Sz, I2, Sz, I2) +
            kron4(I2, Sz, I2, Sz)
        )
        np.testing.assert_allclose(mat, ref, atol=1e-12)

    def test_helper_can_emit_complex_tensors(self):
        N = 4
        H_real = heisenberg_mpo(N)
        H_cplx = heisenberg_mpo(N, dtype=complex)

        for p in range(N):
            self.assertFalse(np.iscomplexobj(H_real[p].get_block().numpy()))
            self.assertTrue(np.iscomplexobj(H_cplx[p].get_block().numpy()))


# ---------------------------------------------------------------------------
# QN MPO correctness
# ---------------------------------------------------------------------------

@SKIP
class TestAutoMPOQN(unittest.TestCase):

    def _heisenberg_ampo_qn(self, N, J=1.0, delta=1.0, h=0.0):
        site = spin_half(qn="Sz")
        ampo = AutoMPO(N, site)
        for i in range(N - 1):
            ampo.add(J * delta, "Sz", i, "Sz", i + 1)
            ampo.add(J / 2,     "Sp", i, "Sm", i + 1)
            ampo.add(J / 2,     "Sm", i, "Sp", i + 1)
        for i in range(N):
            ampo.add(h, "Sz", i)
        return ampo.to_mpo()

    def test_mpo_is_blockform(self):
        H = self._heisenberg_ampo_qn(4)
        for p in range(len(H)):
            self.assertTrue(H[p].is_blockform(),
                            f"Site {p} is not block form")

    def test_matrix_matches_dense_version(self):
        """QN MPO and dense MPO should give the same full matrix."""
        N = 4
        site_dense = spin_half()
        site_qn    = spin_half(qn="Sz")
        ampo_dense = AutoMPO(N, site_dense)
        ampo_qn    = AutoMPO(N, site_qn)
        for i in range(N - 1):
            ampo_dense.add(1.0, "Sz", i, "Sz", i + 1)
            ampo_dense.add(0.5, "Sp", i, "Sm", i + 1)
            ampo_dense.add(0.5, "Sm", i, "Sp", i + 1)
            ampo_qn.add(1.0,   "Sz", i, "Sz", i + 1)
            ampo_qn.add(0.5,   "Sp", i, "Sm", i + 1)
            ampo_qn.add(0.5,   "Sm", i, "Sp", i + 1)
        H_dense = ampo_dense.to_mpo()
        H_qn    = ampo_qn.to_mpo()
        mat_dense = mpo_full_matrix(H_dense)
        mat_qn    = mpo_full_matrix(H_qn)
        np.testing.assert_allclose(mat_qn, mat_dense, atol=1e-12,
                                   err_msg="QN MPO ≠ dense MPO")


# ---------------------------------------------------------------------------
# Charge-changing MPO (Σ Sp_i)
# ---------------------------------------------------------------------------

@SKIP
class TestAutoMPOChargeChanging(unittest.TestCase):

    def test_sum_sp_dense(self):
        """Σ_i S+_i should be a matrix that raises total Sz by 1."""
        N = 3
        site = spin_half()
        ampo = AutoMPO(N, site)
        for i in range(N):
            ampo.add(1.0, "Sp", i)
        H = ampo.to_mpo()
        mat = mpo_full_matrix(H)
        # Build reference: Sp⊗I⊗I + I⊗Sp⊗I + I⊗I⊗Sp
        Sp = np.array([[0, 0], [1, 0]], dtype=float)
        I2 = np.eye(2)
        ref = (np.kron(np.kron(Sp, I2), I2) +
               np.kron(np.kron(I2, Sp), I2) +
               np.kron(np.kron(I2, I2), Sp))
        np.testing.assert_allclose(mat, ref, atol=1e-12)

    def test_sum_sp_qn(self):
        """QN AutoMPO for Σ S+_i: total_charge = +1."""
        N = 3
        site = spin_half(qn="Sz")
        ampo = AutoMPO(N, site)
        for i in range(N):
            ampo.add(1.0, "Sp", i)
        H = ampo.to_mpo()   # should not raise
        self.assertEqual(len(H), N)


# ---------------------------------------------------------------------------
# Dtype decision rules
# ---------------------------------------------------------------------------

@SKIP
class TestAutoMPODtypeRules(unittest.TestCase):

    def _build_simple_ampo(self, site, coeff, op_name):
        ampo = AutoMPO(2, site)
        ampo.add(coeff, op_name, 0)
        return ampo

    def test_real_operator_real_coefficient_gives_real_mpo(self):
        site = spin_half()
        H = self._build_simple_ampo(site, 1.0, "Sz").to_mpo()
        self.assertFalse(np.iscomplexobj(H[0].get_block().numpy()))
        self.assertFalse(np.iscomplexobj(H[1].get_block().numpy()))

    def test_real_operator_complex_coefficient_gives_complex_mpo(self):
        site = spin_half()
        H = self._build_simple_ampo(site, 1.0 + 0.5j, "Sz").to_mpo()
        self.assertTrue(np.iscomplexobj(H[0].get_block().numpy()))
        self.assertTrue(np.iscomplexobj(H[1].get_block().numpy()))

    def test_complex_operator_real_coefficient_gives_complex_mpo(self):
        site = spin_half()
        site.register_op("Cz", np.array([[1j, 0.0], [0.0, -1j]], dtype=complex), 0)
        H = self._build_simple_ampo(site, 2.0, "Cz").to_mpo()
        self.assertTrue(np.iscomplexobj(H[0].get_block().numpy()))
        self.assertTrue(np.iscomplexobj(H[1].get_block().numpy()))

    def test_complex_operator_complex_coefficient_gives_complex_mpo(self):
        site = spin_half()
        site.register_op("Cz", np.array([[1j, 0.0], [0.0, -1j]], dtype=complex), 0)
        H = self._build_simple_ampo(site, 2.0 - 0.3j, "Cz").to_mpo()
        self.assertTrue(np.iscomplexobj(H[0].get_block().numpy()))
        self.assertTrue(np.iscomplexobj(H[1].get_block().numpy()))

    def test_qn_complex_coefficient_gives_complex_mpo(self):
        site = spin_half(qn="Sz")
        ampo = AutoMPO(2, site)
        ampo.add(1.0 + 0.5j, "Sz", 0)
        H = ampo.to_mpo()
        self.assertIn("Complex", H[0].dtype_str())
        self.assertIn("Complex", H[1].dtype_str())


if __name__ == "__main__":
    unittest.main()
