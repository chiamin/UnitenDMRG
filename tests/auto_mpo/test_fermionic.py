"""Tests for fermionic AutoMPO: Jordan-Wigner strings, anti-commutation, edge cases."""

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
    from MPS.physical_sites.site import PhysicalSite
    from MPS.auto_mpo import AutoMPO
    from tests.helpers.mpo_utils import mpo_full_matrix

SKIP = unittest.skipIf(cytnx is None, "cytnx not available")


# ---------------------------------------------------------------------------
# Helper: exact tight-binding matrix
# ---------------------------------------------------------------------------

def tight_binding_exact(N, t):
    """Build the exact d^N x d^N tight-binding Hamiltonian matrix.

    H = -t sum_i (c+_i c_{i+1} + h.c.)

    Uses Jordan-Wigner in the occupation basis:
        |n_0, n_1, ..., n_{N-1}>, n_i in {0, 1}
    Flat index: sum_i n_i * 2^(N-1-i)
    """
    dim = 2**N
    H = np.zeros((dim, dim), dtype=float)

    for state in range(dim):
        bits = [(state >> (N - 1 - i)) & 1 for i in range(N)]
        for i in range(N - 1):
            j = i + 1
            if bits[j] == 1 and bits[i] == 0:
                new_bits = list(bits)
                new_bits[j] = 0
                new_bits[i] = 1
                new_state = sum(new_bits[k] << (N - 1 - k) for k in range(N))
                sign = (-1) ** sum(bits[k] for k in range(i + 1, j))
                H[new_state, state] += -t * sign
                H[state, new_state] += -t * sign

    return H


# ---------------------------------------------------------------------------
# Tight-binding model (dense)
# ---------------------------------------------------------------------------

@SKIP
class TestFermionicAutoMPODense(unittest.TestCase):
    """Test AutoMPO with JW string for dense spinless fermion."""

    def _tight_binding_mpo(self, N, t=1.0):
        site = spinless_fermion()
        ampo = AutoMPO(N, site)
        for i in range(N - 1):
            ampo.add(-t, "Cdag", i, "C", i + 1)
            ampo.add(-t, "Cdag", i + 1, "C", i)
        return ampo.to_mpo()

    def test_nearest_neighbor_N3(self):
        N, t = 3, 1.0
        H_mpo = self._tight_binding_mpo(N, t)
        mat_mpo = mpo_full_matrix(H_mpo)
        mat_exact = tight_binding_exact(N, t)
        np.testing.assert_allclose(mat_mpo, mat_exact, atol=1e-12)

    def test_nearest_neighbor_N4(self):
        N, t = 4, 1.0
        H_mpo = self._tight_binding_mpo(N, t)
        mat_mpo = mpo_full_matrix(H_mpo)
        mat_exact = tight_binding_exact(N, t)
        np.testing.assert_allclose(mat_mpo, mat_exact, atol=1e-12)

    def test_nearest_neighbor_N5(self):
        N, t = 5, 0.7
        H_mpo = self._tight_binding_mpo(N, t)
        mat_mpo = mpo_full_matrix(H_mpo)
        mat_exact = tight_binding_exact(N, t)
        np.testing.assert_allclose(mat_mpo, mat_exact, atol=1e-12)

    def test_hermitian(self):
        N = 4
        mat = mpo_full_matrix(self._tight_binding_mpo(N))
        np.testing.assert_allclose(mat, mat.T, atol=1e-12)

    def test_non_adjacent_hopping(self):
        """c+_0 c_2 + h.c. -- must insert F at site 1."""
        N = 4
        site = spinless_fermion()
        ampo = AutoMPO(N, site)
        ampo.add(-1.0, "Cdag", 0, "C", 2)
        ampo.add(-1.0, "Cdag", 2, "C", 0)
        H_mpo = mpo_full_matrix(ampo.to_mpo())

        dim = 2**N
        H_exact = np.zeros((dim, dim), dtype=float)
        for state in range(dim):
            bits = [(state >> (N - 1 - k)) & 1 for k in range(N)]
            if bits[2] == 1 and bits[0] == 0:
                new_bits = list(bits)
                new_bits[2] = 0
                new_bits[0] = 1
                new_state = sum(new_bits[k] << (N - 1 - k) for k in range(N))
                sign = (-1) ** bits[1]
                H_exact[new_state, state] += -1.0 * sign
                H_exact[state, new_state] += -1.0 * sign

        np.testing.assert_allclose(H_mpo, H_exact, atol=1e-12)


# ---------------------------------------------------------------------------
# Tight-binding model (QN)
# ---------------------------------------------------------------------------

@SKIP
class TestFermionicAutoMPOQN(unittest.TestCase):
    """Test AutoMPO with JW string for U(1) QN spinless fermion."""

    def _tight_binding_mpo(self, N, t=1.0, qn="N"):
        site = spinless_fermion(qn=qn)
        ampo = AutoMPO(N, site)
        for i in range(N - 1):
            ampo.add(-t, "Cdag", i, "C", i + 1)
            ampo.add(-t, "Cdag", i + 1, "C", i)
        return ampo.to_mpo()

    def test_qn_matches_dense_N4(self):
        """QN MPO and dense MPO must give the same full matrix."""
        N, t = 4, 1.0
        H_dense = mpo_full_matrix(self._tight_binding_mpo(N, t, qn=None))
        H_qn = mpo_full_matrix(self._tight_binding_mpo(N, t, qn="N"))
        np.testing.assert_allclose(H_qn, H_dense, atol=1e-12)

    def test_qn_is_blockform(self):
        H = self._tight_binding_mpo(4, qn="N")
        for p in range(len(H)):
            self.assertTrue(H[p].is_blockform())

    def test_parity_matches_dense_N4(self):
        """Z2 parity MPO and dense MPO must give the same full matrix."""
        N, t = 4, 1.0
        H_dense = mpo_full_matrix(self._tight_binding_mpo(N, t, qn=None))
        H_z2 = mpo_full_matrix(self._tight_binding_mpo(N, t, qn="parity"))
        np.testing.assert_allclose(H_z2, H_dense, atol=1e-12)

    def test_parity_is_blockform(self):
        H = self._tight_binding_mpo(4, qn="parity")
        for p in range(len(H)):
            self.assertTrue(H[p].is_blockform())


# ---------------------------------------------------------------------------
# Anti-commutation: Cdag_i C_j = -C_j Cdag_i
# ---------------------------------------------------------------------------

@SKIP
class TestFermionicAntiCommutation(unittest.TestCase):
    """Verify Cdag_i C_j = -C_j Cdag_i for both i<j and i>j.

    Build two single-term MPOs and compare their full dense matrices.
    The anti-commutation relation must hold regardless of operator ordering.
    """

    def _single_term_matrix(self, N, coeff, op1, s1, op2, s2):
        """Build the full matrix for a single term: coeff * op1_{s1} op2_{s2}."""
        site = spinless_fermion()
        ampo = AutoMPO(N, site)
        ampo.add(coeff, op1, s1, op2, s2)
        return mpo_full_matrix(ampo.to_mpo())

    def test_anticommutation_adjacent_forward(self):
        """Cdag_0 C_1 = -C_1 Cdag_0  (i<j)."""
        N = 4
        H_CdagC = self._single_term_matrix(N, 1.0, "Cdag", 0, "C", 1)
        H_CCdag = self._single_term_matrix(N, 1.0, "C", 1, "Cdag", 0)
        np.testing.assert_allclose(H_CdagC, -H_CCdag, atol=1e-12,
                                   err_msg="Cdag_0 C_1 != -C_1 Cdag_0")

    def test_anticommutation_adjacent_backward(self):
        """Cdag_1 C_0 = -C_0 Cdag_1  (i>j)."""
        N = 4
        H_CdagC = self._single_term_matrix(N, 1.0, "Cdag", 1, "C", 0)
        H_CCdag = self._single_term_matrix(N, 1.0, "C", 0, "Cdag", 1)
        np.testing.assert_allclose(H_CdagC, -H_CCdag, atol=1e-12,
                                   err_msg="Cdag_1 C_0 != -C_0 Cdag_1")

    def test_anticommutation_non_adjacent_forward(self):
        """Cdag_0 C_3 = -C_3 Cdag_0  (i<j, non-adjacent)."""
        N = 5
        H_CdagC = self._single_term_matrix(N, 1.0, "Cdag", 0, "C", 3)
        H_CCdag = self._single_term_matrix(N, 1.0, "C", 3, "Cdag", 0)
        np.testing.assert_allclose(H_CdagC, -H_CCdag, atol=1e-12,
                                   err_msg="Cdag_0 C_3 != -C_3 Cdag_0")

    def test_anticommutation_non_adjacent_backward(self):
        """Cdag_3 C_0 = -C_0 Cdag_3  (i>j, non-adjacent)."""
        N = 5
        H_CdagC = self._single_term_matrix(N, 1.0, "Cdag", 3, "C", 0)
        H_CCdag = self._single_term_matrix(N, 1.0, "C", 0, "Cdag", 3)
        np.testing.assert_allclose(H_CdagC, -H_CCdag, atol=1e-12,
                                   err_msg="Cdag_3 C_0 != -C_0 Cdag_3")

    def test_anticommutation_middle_sites(self):
        """Cdag_1 C_3 = -C_3 Cdag_1  (both in the interior)."""
        N = 5
        H_CdagC = self._single_term_matrix(N, 1.0, "Cdag", 1, "C", 3)
        H_CCdag = self._single_term_matrix(N, 1.0, "C", 3, "Cdag", 1)
        np.testing.assert_allclose(H_CdagC, -H_CCdag, atol=1e-12,
                                   err_msg="Cdag_1 C_3 != -C_3 Cdag_1")


@SKIP
class TestFermionicAntiCommutationQN(unittest.TestCase):
    """QN version of anti-commutation tests.

    Verify that QN MPO gives the same matrix as dense for both orderings.
    """

    def _single_term_matrix(self, N, coeff, op1, s1, op2, s2, qn="N"):
        site = spinless_fermion(qn=qn)
        ampo = AutoMPO(N, site)
        ampo.add(coeff, op1, s1, op2, s2)
        return mpo_full_matrix(ampo.to_mpo())

    def _single_term_matrix_dense(self, N, coeff, op1, s1, op2, s2):
        site = spinless_fermion()
        ampo = AutoMPO(N, site)
        ampo.add(coeff, op1, s1, op2, s2)
        return mpo_full_matrix(ampo.to_mpo())

    def _check(self, N, op1, s1, op2, s2, qn="N"):
        H_qn = self._single_term_matrix(N, 1.0, op1, s1, op2, s2, qn=qn)
        H_dense = self._single_term_matrix_dense(N, 1.0, op1, s1, op2, s2)
        np.testing.assert_allclose(H_qn, H_dense, atol=1e-12,
                                   err_msg=f"QN != dense for {op1}_{s1} {op2}_{s2}")

    def test_u1_adjacent_forward(self):
        self._check(4, "Cdag", 0, "C", 1)

    def test_u1_adjacent_backward(self):
        self._check(4, "C", 1, "Cdag", 0)

    def test_u1_non_adjacent(self):
        self._check(5, "Cdag", 0, "C", 3)

    def test_u1_non_adjacent_backward(self):
        self._check(5, "C", 3, "Cdag", 0)

    def test_z2_adjacent_forward(self):
        self._check(4, "Cdag", 0, "C", 1, qn="parity")

    def test_z2_non_adjacent_backward(self):
        self._check(5, "C", 3, "Cdag", 0, qn="parity")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@SKIP
class TestFermionicAutoMPOEdgeCases(unittest.TestCase):

    def test_missing_F_raises(self):
        """A site with fermionic ops but no F operator should fail."""
        bond = cytnx.Bond(2, cytnx.BD_IN)
        site = PhysicalSite(bond, type_name="BadFermion")
        site.register_op("I", np.eye(2), 0)
        site.register_op("C", np.array([[0, 1], [0, 0]], dtype=float), 0, fermionic=True)
        site.register_op("Cdag", np.array([[0, 0], [1, 0]], dtype=float), 0, fermionic=True)
        ampo = AutoMPO(3, site)
        with self.assertRaises(KeyError):
            ampo.add(-1.0, "Cdag", 0, "C", 1)

    def test_bosonic_terms_unaffected(self):
        """Adding bosonic (N_i N_j) terms alongside hopping should not break."""
        N = 4
        site = spinless_fermion()
        ampo = AutoMPO(N, site)
        for i in range(N - 1):
            ampo.add(-1.0, "Cdag", i, "C", i + 1)
            ampo.add(-1.0, "Cdag", i + 1, "C", i)
            ampo.add(2.0, "N", i, "N", i + 1)
        H_mpo = mpo_full_matrix(ampo.to_mpo())

        H_hop = tight_binding_exact(N, 1.0)
        dim = 2**N
        H_nn = np.zeros((dim, dim), dtype=float)
        Nop = np.diag([0.0, 1.0])
        I2 = np.eye(2)
        for i in range(N - 1):
            ops = [I2] * N
            ops[i] = Nop
            ops[i + 1] = Nop
            term = ops[0]
            for k in range(1, N):
                term = np.kron(term, ops[k])
            H_nn += 2.0 * term

        np.testing.assert_allclose(H_mpo, H_hop + H_nn, atol=1e-12)


# ---------------------------------------------------------------------------
# 2D tight-binding on a square lattice (long-range hopping in 1D mapping)
# ---------------------------------------------------------------------------

def _square_tb_exact_energies(Lx, Ly, t):
    """Analytical single-particle energies for 2D tight-binding on OBC square lattice."""
    energies = []
    for nx in range(1, Lx + 1):
        for ny in range(1, Ly + 1):
            e = -2.0 * t * (np.cos(np.pi * nx / (Lx + 1))
                            + np.cos(np.pi * ny / (Ly + 1)))
            energies.append(e)
    return sorted(energies)


@SKIP
class TestFermionicAutoMPO2D(unittest.TestCase):
    """2D tight-binding on a square lattice via AutoMPO.

    The y-direction hoppings are long-range in the 1D row-major mapping
    (distance Lx), exercising JW strings across multiple sites.
    """

    def _build_2d_tb_mpo(self, Lx, Ly, t=1.0):
        from lattice import SquareLattice
        lat = SquareLattice(Lx, Ly)
        site = spinless_fermion()
        ampo = AutoMPO(lat.N, site)
        for i, j in lat.bonds():
            ampo.add(-t, "Cdag", i, "C", j)
            ampo.add(-t, "Cdag", j, "C", i)
        return ampo.to_mpo()

    def test_gs_energy_3x2_half_filling(self):
        """Ground state energy of 3x2 tight-binding at half filling."""
        Lx, Ly, t = 3, 2, 1.0
        Nf = 3
        H_mat = mpo_full_matrix(self._build_2d_tb_mpo(Lx, Ly, t))
        evals = np.linalg.eigvalsh(H_mat)
        sp_energies = _square_tb_exact_energies(Lx, Ly, t)
        E0_exact = sum(sp_energies[:Nf])
        np.testing.assert_allclose(evals[0], E0_exact, atol=1e-10)

    def test_gs_energy_3x3(self):
        """Ground state energy of 3x3 tight-binding (global GS)."""
        Lx, Ly, t = 3, 3, 0.8
        N = Lx * Ly
        H_mat = mpo_full_matrix(self._build_2d_tb_mpo(Lx, Ly, t))
        evals = np.linalg.eigvalsh(H_mat)
        sp = _square_tb_exact_energies(Lx, Ly, t)
        E0_exact = min(sum(sp[:Nf]) for Nf in range(N + 1))
        np.testing.assert_allclose(evals[0], E0_exact, atol=1e-10)

    def test_full_spectrum_vs_ed_3x2(self):
        """Full MPO spectrum matches exact diagonalization for 3x2."""
        Lx, Ly, t = 3, 2, 1.0
        N = Lx * Ly
        dim = 2**N
        from lattice import SquareLattice
        lat = SquareLattice(Lx, Ly)

        H_ed = np.zeros((dim, dim))
        for state in range(dim):
            bits = [(state >> (N - 1 - k)) & 1 for k in range(N)]
            for i, j in lat.bonds():
                if bits[j] == 1 and bits[i] == 0:
                    new_bits = list(bits)
                    new_bits[j] = 0
                    new_bits[i] = 1
                    new_state = sum(new_bits[k] << (N - 1 - k) for k in range(N))
                    sign = (-1) ** sum(bits[k] for k in range(i + 1, j))
                    H_ed[new_state, state] += -t * sign
                    H_ed[state, new_state] += -t * sign

        H_mpo = mpo_full_matrix(self._build_2d_tb_mpo(Lx, Ly, t))
        np.testing.assert_allclose(H_mpo, H_ed, atol=1e-12)

    def test_full_spectrum_vs_ed_3x3(self):
        """Full MPO spectrum matches exact diagonalization for 3x3."""
        Lx, Ly, t = 3, 3, 1.0
        from lattice import SquareLattice
        lat = SquareLattice(Lx, Ly)
        H_mpo = mpo_full_matrix(self._build_2d_tb_mpo(Lx, Ly, t))
        H_ed = _tb_ed_matrix(lat, t)
        np.testing.assert_allclose(H_mpo, H_ed, atol=1e-12)

    def test_hermitian(self):
        H_mat = mpo_full_matrix(self._build_2d_tb_mpo(3, 2))
        np.testing.assert_allclose(H_mat, H_mat.T, atol=1e-12)


# ---------------------------------------------------------------------------
# Boundary condition tests: OBC / PBC combinations
# ---------------------------------------------------------------------------

def _tb_ed_matrix(lat, t):
    """Build exact tight-binding matrix with explicit JW for any lattice."""
    N = lat.N
    dim = 2**N
    H = np.zeros((dim, dim))
    for state in range(dim):
        bits = [(state >> (N - 1 - k)) & 1 for k in range(N)]
        for i, j in lat.bonds():
            if bits[j] == 1 and bits[i] == 0:
                new_bits = list(bits)
                new_bits[j] = 0
                new_bits[i] = 1
                new_state = sum(new_bits[k] << (N - 1 - k) for k in range(N))
                sign = (-1) ** sum(bits[k] for k in range(i + 1, j))
                H[new_state, state] += -t * sign
                H[state, new_state] += -t * sign
    return H


def _tb_mpo_matrix(lat, t, qn=None):
    """Build tight-binding MPO matrix for any lattice."""
    site = spinless_fermion(qn=qn)
    ampo = AutoMPO(lat.N, site)
    for i, j in lat.bonds():
        ampo.add(-t, "Cdag", i, "C", j)
        ampo.add(-t, "Cdag", j, "C", i)
    return mpo_full_matrix(ampo.to_mpo())


@SKIP
class TestFermionicAutoMPOBoundaryConditions(unittest.TestCase):
    """Test all four boundary condition combinations (dense).

    Compare MPO full matrix against exact diagonalization with explicit JW.
    """

    Lx, Ly, t = 3, 2, 1.0

    def _check_bc(self, xpbc, ypbc):
        from lattice import SquareLattice
        lat = SquareLattice(self.Lx, self.Ly, xpbc=xpbc, ypbc=ypbc)
        H_mpo = _tb_mpo_matrix(lat, self.t)
        H_ed = _tb_ed_matrix(lat, self.t)
        np.testing.assert_allclose(H_mpo, H_ed, atol=1e-12,
                                   err_msg=f"Failed for xpbc={xpbc}, ypbc={ypbc}")
        np.testing.assert_allclose(H_mpo, H_mpo.T, atol=1e-12,
                                   err_msg=f"Not Hermitian for xpbc={xpbc}, ypbc={ypbc}")

    def test_obc_obc(self):
        self._check_bc(xpbc=False, ypbc=False)

    def test_xpbc_obc(self):
        self._check_bc(xpbc=True, ypbc=False)

    def test_obc_ypbc(self):
        self._check_bc(xpbc=False, ypbc=True)

    def test_xpbc_ypbc(self):
        self._check_bc(xpbc=True, ypbc=True)


@SKIP
class TestFermionicAutoMPOBoundaryConditionsQN(unittest.TestCase):
    """Test all four boundary condition combinations (U1 and Z2 QN).

    QN MPO must give the same full matrix as the dense version.
    """

    Lx, Ly, t = 3, 2, 1.0

    def _check_bc_qn(self, xpbc, ypbc, qn):
        from lattice import SquareLattice
        lat = SquareLattice(self.Lx, self.Ly, xpbc=xpbc, ypbc=ypbc)
        H_dense = _tb_mpo_matrix(lat, self.t, qn=None)
        H_qn = _tb_mpo_matrix(lat, self.t, qn=qn)
        np.testing.assert_allclose(
            H_qn, H_dense, atol=1e-12,
            err_msg=f"QN({qn}) != dense for xpbc={xpbc}, ypbc={ypbc}")

    def test_u1_obc_obc(self):
        self._check_bc_qn(False, False, "N")

    def test_u1_xpbc_obc(self):
        self._check_bc_qn(True, False, "N")

    def test_u1_obc_ypbc(self):
        self._check_bc_qn(False, True, "N")

    def test_u1_xpbc_ypbc(self):
        self._check_bc_qn(True, True, "N")

    def test_z2_obc_obc(self):
        self._check_bc_qn(False, False, "parity")

    def test_z2_xpbc_obc(self):
        self._check_bc_qn(True, False, "parity")

    def test_z2_obc_ypbc(self):
        self._check_bc_qn(False, True, "parity")

    def test_z2_xpbc_ypbc(self):
        self._check_bc_qn(True, True, "parity")


if __name__ == "__main__":
    unittest.main()
