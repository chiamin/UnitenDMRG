"""Tests for Hubbard model via AutoMPO with electron site."""

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
    from MPS.auto_mpo import AutoMPO
    from tests.helpers.mpo_utils import mpo_full_matrix

SKIP = unittest.skipIf(cytnx is None, "cytnx not available")


# ---------------------------------------------------------------------------
# Helper: exact Hubbard matrix via ED
# ---------------------------------------------------------------------------

def hubbard_ed(N, t, U):
    """Build the exact 4^N x 4^N Hubbard Hamiltonian.

    H = -t sum_{i,sigma} (c†_{i,sigma} c_{i+1,sigma} + h.c.)
        + U sum_i n_{i,up} n_{i,dn}

    Basis: |s_0, s_1, ..., s_{N-1}> where s_i in {0,1,2,3}
    maps to (n_up, n_dn) = (0,0), (1,0), (0,1), (1,1).
    Flat index: sum_i s_i * 4^(N-1-i).

    Jordan-Wigner: fermions are ordered as up_0, dn_0, up_1, dn_1, ...
    so the JW index of c_{i,up} is 2*i and c_{i,dn} is 2*i+1.
    """
    dim = 4**N
    H = np.zeros((dim, dim), dtype=float)

    def bits(state):
        """Return list of (n_up, n_dn) for each site."""
        occ = []
        for i in range(N):
            s = (state // (4**(N - 1 - i))) % 4
            nup = 1 if s in (1, 3) else 0
            ndn = 1 if s in (2, 3) else 0
            occ.append((nup, ndn))
        return occ

    def flat_index(occ):
        idx = 0
        for i, (nup, ndn) in enumerate(occ):
            s = nup + 2 * ndn   # 0,1,2,3
            idx += s * 4**(N - 1 - i)
        return idx

    def jw_sign(occ, site, sigma):
        """JW sign for annihilating sigma at site.

        Fermion ordering: up_0, dn_0, up_1, dn_1, ...
        JW index of c_{site,sigma}: 2*site + (0 if up, 1 if dn).
        Sign = (-1)^(number of occupied fermion modes with index < jw_idx).
        """
        jw_idx = 2 * site + (0 if sigma == "up" else 1)
        count = 0
        for i in range(N):
            nup, ndn = occ[i]
            if 2 * i < jw_idx:
                count += nup
            if 2 * i + 1 < jw_idx:
                count += ndn
        return (-1) ** count

    for state in range(dim):
        occ = bits(state)

        # U term: U * n_up * n_dn at each site
        for i in range(N):
            nup, ndn = occ[i]
            if nup == 1 and ndn == 1:
                H[state, state] += U

        # Hopping: -t * c†_{i,sigma} c_{i+1,sigma} + h.c.
        for i in range(N - 1):
            j = i + 1
            for sigma in ["up", "dn"]:
                s_idx = 0 if sigma == "up" else 1
                ni = occ[i][s_idx]
                nj = occ[j][s_idx]

                # c†_i c_j: annihilate sigma at j, create at i
                if nj == 1 and ni == 0:
                    new_occ = [list(o) for o in occ]
                    # Annihilate at j
                    sign_j = jw_sign(occ, j, sigma)
                    new_occ[j][s_idx] = 0
                    # Create at i
                    sign_i = jw_sign(new_occ, i, sigma)
                    new_occ[i][s_idx] = 1
                    new_occ_tuples = [tuple(o) for o in new_occ]
                    new_state = flat_index(new_occ_tuples)
                    sign = sign_i * sign_j
                    H[new_state, state] += -t * sign
                    H[state, new_state] += -t * sign  # h.c.

    return H


def hubbard_mpo_matrix(N, t, U, qn=None):
    """Build Hubbard MPO and return full matrix."""
    site = electron(qn=qn)
    ampo = AutoMPO(N, site)
    for i in range(N - 1):
        for op_dag, op in [("Cupdag", "Cup"), ("Cdndag", "Cdn")]:
            ampo.add(-t, op_dag, i, op, i + 1)
            ampo.add(-t, op_dag, i + 1, op, i)
    for i in range(N):
        ampo.add(U, "Nup", i, "Ndn", i)
    return mpo_full_matrix(ampo.to_mpo())


# ---------------------------------------------------------------------------
# Dense Hubbard tests
# ---------------------------------------------------------------------------

@SKIP
class TestHubbardDense(unittest.TestCase):
    """Compare dense Hubbard MPO against exact diagonalization."""

    def test_N3_vs_ed(self):
        N, t, U = 3, 1.0, 2.0
        H_mpo = hubbard_mpo_matrix(N, t, U)
        H_ed = hubbard_ed(N, t, U)
        np.testing.assert_allclose(H_mpo, H_ed, atol=1e-12)

    def test_N4_vs_ed(self):
        N, t, U = 4, 1.0, 4.0
        H_mpo = hubbard_mpo_matrix(N, t, U)
        H_ed = hubbard_ed(N, t, U)
        np.testing.assert_allclose(H_mpo, H_ed, atol=1e-12)

    def test_hermitian(self):
        N, t, U = 3, 1.0, 2.0
        H = hubbard_mpo_matrix(N, t, U)
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_U_only(self):
        """Pure interaction, no hopping."""
        N, U = 3, 3.0
        H_mpo = hubbard_mpo_matrix(N, t=0.0, U=U)
        H_ed = hubbard_ed(N, t=0.0, U=U)
        np.testing.assert_allclose(H_mpo, H_ed, atol=1e-12)

    def test_t_only(self):
        """Pure hopping, no interaction."""
        N, t = 3, 1.5
        H_mpo = hubbard_mpo_matrix(N, t=t, U=0.0)
        H_ed = hubbard_ed(N, t=t, U=0.0)
        np.testing.assert_allclose(H_mpo, H_ed, atol=1e-12)


# ---------------------------------------------------------------------------
# QN Hubbard tests
# ---------------------------------------------------------------------------

@SKIP
class TestHubbardQN(unittest.TestCase):
    """QN Hubbard MPO must match the dense version."""

    N, t, U = 3, 1.0, 2.0

    def _check_qn(self, qn):
        H_dense = hubbard_mpo_matrix(self.N, self.t, self.U, qn=None)
        H_qn = hubbard_mpo_matrix(self.N, self.t, self.U, qn=qn)
        np.testing.assert_allclose(H_qn, H_dense, atol=1e-12,
                                   err_msg=f"QN({qn}) != dense")

    def test_ntot(self):
        self._check_qn("Ntot")

    def test_sz(self):
        self._check_qn("Sz")

    def test_ntot_sz(self):
        self._check_qn("Ntot,Sz")

    def test_nup_ndn(self):
        self._check_qn("Nup,Ndn")

    def test_blockform(self):
        """All QN MPOs should be in block form."""
        for qn in ["Ntot", "Sz", "Ntot,Sz", "Nup,Ndn"]:
            site = electron(qn=qn)
            ampo = AutoMPO(self.N, site)
            for i in range(self.N - 1):
                for od, o in [("Cupdag", "Cup"), ("Cdndag", "Cdn")]:
                    ampo.add(-self.t, od, i, o, i + 1)
                    ampo.add(-self.t, od, i + 1, o, i)
            for i in range(self.N):
                ampo.add(self.U, "Nup", i, "Ndn", i)
            H = ampo.to_mpo()
            for p in range(len(H)):
                self.assertTrue(H[p].is_blockform(),
                                f"qn={qn}, site {p} not blockform")


if __name__ == "__main__":
    unittest.main()
