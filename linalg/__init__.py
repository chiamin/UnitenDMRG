"""Type-agnostic linear algebra routines for DMRG and related algorithms.

All routines operate on Cytnx UniTensor vectors, but avoid any numpy array
conversion in the hot path.  The only assumption is that the vector type
supports the following operations via duck typing:

    v + w, v - w         : vector addition / subtraction
    a * v, v * a         : scalar multiplication  (a is a Python scalar)
    v.clone()            : deep copy
    v.Norm().item()      : Euclidean norm as a Python float
    v * 0.               : zero vector of the same structure

The inner product is provided by the module-level `inner` function, which
is the only UniTensor-specific operation.
"""

from .inner import inner
from .lanczos import lanczos, lanczos_expm_multiply
from .davidson import davidson

__all__ = [
    "inner",
    "lanczos",
    "lanczos_expm_multiply",
    "davidson",
]
