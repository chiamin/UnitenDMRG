"""Helpers for linalg solver tests.

Wraps a numpy matrix M as an `apply(v) -> M @ v` callable acting on
rank-1 UniTensor vectors with label ['i'].  This lets the solver run on
genuine UniTensor objects (so label-restoration and `inner`/`Dagger`
paths are exercised) while the reference answer comes from numpy.
"""

from __future__ import annotations

import numpy as np
import cytnx


def vec(arr: np.ndarray) -> "cytnx.UniTensor":
    """Wrap a 1-D numpy array as a rank-1 UniTensor with label ['i']."""
    u = cytnx.UniTensor(cytnx.from_numpy(np.ascontiguousarray(arr)), rowrank=1)
    u.set_labels(["i"])
    return u


def to_np(v: "cytnx.UniTensor") -> np.ndarray:
    return v.get_block().numpy().ravel()


def make_apply(M: np.ndarray):
    """Return apply(v) = M @ v as a UniTensor->UniTensor callable.

    The result preserves the input dtype family (complex stays complex).
    """
    def apply(v: "cytnx.UniTensor") -> "cytnx.UniTensor":
        x = to_np(v)
        return vec(M @ x)
    return apply
