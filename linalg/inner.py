"""Inner product for UniTensor vectors."""

from __future__ import annotations

import cytnx


def inner(v1: "cytnx.UniTensor", v2: "cytnx.UniTensor") -> complex:
    """Compute the inner product <v1|v2> for two UniTensors.

    Both tensors must have identical label sets and compatible bond directions
    (v1 acts as a bra, v2 as a ket).  Contracts all shared indices and returns
    a Python scalar.

    Notes
    -----
    Dagger() on v1 flips bond directions and complex-conjugates elements so
    that the result is the standard Hilbert-space inner product.
    """
    #
    #  <v1|v2> :  v1†──O──v2  →  scalar
    #
    return cytnx.Contract(v1.Dagger(), v2).item()
