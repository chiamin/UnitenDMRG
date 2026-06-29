"""Unit tests for linalg.inner (dense and QN, real and complex).

`inner(v1, v2) = <v1|v2> = Contract(v1.Dagger(), v2).item()`.

Coverage
--------
Dense and QN tensors are tested independently (a bug in QN block-tensor
contraction does not surface in dense tests, and vice versa).  For each:

- real:    <v|v> = ||v||^2,  symmetry <v|w> = <w|v>,  linearity in 2nd arg
- complex: <v|v> real & positive,  conjugate symmetry <v|w> = conj(<w|v>),
           sesquilinearity (linear in 2nd arg, antilinear in 1st arg)

The complex cases are the only ones that can detect a missing .Dagger() on the
bra side, since conj(x) == x for real numbers.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import cytnx
except ImportError:
    cytnx = None

from linalg import inner

from ._helpers import vec, make_qn_vector, qn_to_np


# ===========================================================================
# Dense
# ===========================================================================

@unittest.skipIf(cytnx is None, "cytnx not available")
class TestInnerDenseReal(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        self.v_np = rng.standard_normal(8)
        self.w_np = rng.standard_normal(8)
        self.v = vec(self.v_np)
        self.w = vec(self.w_np)

    def test_self_inner_equals_norm_squared(self):
        result = inner(self.v, self.v)
        expected = float(np.dot(self.v_np.conj(), self.v_np).real)
        self.assertAlmostEqual(result.real, expected, places=10)
        self.assertAlmostEqual(abs(result.imag), 0.0, places=10)

    def test_symmetry(self):
        """For real vectors, <v|w> = <w|v>."""
        self.assertAlmostEqual(inner(self.v, self.w), inner(self.w, self.v),
                               places=10)

    def test_linearity_in_second_arg(self):
        alpha = 2.5
        lhs = inner(self.v, self.w * alpha)
        rhs = alpha * inner(self.v, self.w)
        self.assertAlmostEqual(lhs, rhs, places=10)


@unittest.skipIf(cytnx is None, "cytnx not available")
class TestInnerDenseComplex(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(101)
        self.v_np = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        self.w_np = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        self.alpha = 2.0 + 1.5j
        self.v = vec(self.v_np)
        self.w = vec(self.w_np)

    def test_self_inner_is_real_and_positive(self):
        result = inner(self.v, self.v)
        expected = float(np.dot(self.v_np.conj(), self.v_np).real)
        self.assertAlmostEqual(result.real, expected, places=10)
        self.assertAlmostEqual(result.imag, 0.0, places=10)
        self.assertGreater(result.real, 0.0)

    def test_conjugate_symmetry(self):
        """<v|w> = conj(<w|v>)."""
        vw = complex(inner(self.v, self.w))
        wv = complex(inner(self.w, self.v))
        self.assertAlmostEqual(vw.real,  wv.real,  places=10)
        self.assertAlmostEqual(vw.imag, -wv.imag,  places=10)

    def test_linear_in_second_arg(self):
        lhs = complex(inner(self.v, self.w * self.alpha))
        rhs = self.alpha * complex(inner(self.v, self.w))
        self.assertAlmostEqual(lhs, rhs, places=10)

    def test_antilinear_in_first_arg(self):
        """<alpha*v|w> = conj(alpha) * <v|w>  (the missing-Dagger detector)."""
        lhs = complex(inner(self.v * self.alpha, self.w))
        rhs = self.alpha.conjugate() * complex(inner(self.v, self.w))
        self.assertAlmostEqual(lhs, rhs, places=10)


# ===========================================================================
# QN
# ===========================================================================

L_SEC, L_DEG = [0, 1, 2], [3, 3, 2]
I_SEC, I_DEG = [0, 1], [2, 2]


@unittest.skipIf(cytnx is None, "cytnx not available")
class TestInnerQNReal(unittest.TestCase):

    def setUp(self):
        self.v = make_qn_vector(L_SEC, L_DEG, I_SEC, I_DEG, "real", seed=2)
        self.w = make_qn_vector(L_SEC, L_DEG, I_SEC, I_DEG, "real", seed=3)
        self.v_np = qn_to_np(self.v)
        self.w_np = qn_to_np(self.w)

    def test_self_inner_equals_norm_squared(self):
        result = inner(self.v, self.v)
        expected = float(np.dot(self.v_np.conj(), self.v_np).real)
        self.assertAlmostEqual(result.real, expected, places=10)
        self.assertAlmostEqual(abs(result.imag), 0.0, places=10)

    def test_matches_numpy_dot(self):
        """<v|w> matches the numpy reference on the flattened blocks."""
        result = complex(inner(self.v, self.w))
        expected = np.dot(self.v_np.conj(), self.w_np)
        self.assertAlmostEqual(result, expected, places=10)

    def test_symmetry(self):
        self.assertAlmostEqual(inner(self.v, self.w), inner(self.w, self.v),
                               places=10)


@unittest.skipIf(cytnx is None, "cytnx not available")
class TestInnerQNComplex(unittest.TestCase):

    def setUp(self):
        self.v = make_qn_vector(L_SEC, L_DEG, I_SEC, I_DEG, "complex", seed=2)
        self.w = make_qn_vector(L_SEC, L_DEG, I_SEC, I_DEG, "complex", seed=3)
        self.v_np = qn_to_np(self.v)
        self.w_np = qn_to_np(self.w)

    def test_self_inner_is_real_and_positive(self):
        result = inner(self.v, self.v)
        expected = float(np.dot(self.v_np.conj(), self.v_np).real)
        self.assertAlmostEqual(result.real, expected, places=10)
        self.assertAlmostEqual(result.imag, 0.0, places=10)
        self.assertGreater(result.real, 0.0)

    def test_matches_numpy_dot(self):
        """<v|w> = conj(v)·w on flattened blocks — detects missing Dagger."""
        result = complex(inner(self.v, self.w))
        expected = np.dot(self.v_np.conj(), self.w_np)
        self.assertAlmostEqual(result, expected, places=10)

    def test_conjugate_symmetry(self):
        vw = complex(inner(self.v, self.w))
        wv = complex(inner(self.w, self.v))
        self.assertAlmostEqual(vw.real,  wv.real,  places=10)
        self.assertAlmostEqual(vw.imag, -wv.imag,  places=10)


if __name__ == "__main__":
    unittest.main()
