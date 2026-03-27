"""Type-agnostic linear algebra routines for DMRG and related algorithms.

All routines operate on Cytnx UniTensor vectors, but avoid any numpy array
conversion in the hot path.  The only assumption is that the vector type
supports the following operations via duck typing:

    v + w, v - w         : vector addition / subtraction
    a * v, v * a         : scalar multiplication  (a is a Python scalar)
    v.clone()            : deep copy
    v.Norm().item()      : Euclidean norm as a Python float
    v * 0.               : zero vector of the same structure

The inner product is provided by the module-level ``inner`` function, which
is the only UniTensor-specific operation.
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    import cytnx
except ImportError as exc:
    raise ImportError("cytnx is required for linalg.py.") from exc


# ---------------------------------------------------------------------------
# Inner product
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Lanczos ground-state eigensolver
# ---------------------------------------------------------------------------

def lanczos(apply, v0: "cytnx.UniTensor", k: int = 20) -> tuple:
    """Find the ground-state energy and eigenvector of a Hermitian operator.

    Builds a Krylov basis { apply^i(v0) : i < k } and solves the small
    eigenproblem in that subspace (full re-orthogonalisation).

    Parameters
    ----------
    apply : callable
        apply(v) -> v'  —  the matrix-vector product H|v>.
        v and v' must be the same type as v0.
    v0 : UniTensor
        Initial vector (need not be normalised).
    k : int
        Maximum number of Lanczos iterations.

    Returns
    -------
    E0 : float
        Ground-state energy (lowest eigenvalue).
    psi : UniTensor
        Ground-state vector, normalised to unit norm.
    """
    T, vecs = _lanczos_iterations(apply, v0, k)
    evals, evecs = np.linalg.eigh(T)

    # Back-transform: psi = sum_i evecs[i, 0] * vecs[i]
    # Cytnx addition (a + b) drops labels; restore them after each step.
    _labels = list(v0.labels())
    c = evecs[:, 0]                 # ground-state coefficients in Krylov basis
    psi = c[0] * vecs[0]
    for i in range(1, len(vecs)):
        psi = psi + c[i] * vecs[i]
        psi.set_labels(_labels)

    norm = psi.Norm().item()
    if abs(norm - 1.) > 1.e-6:
        warnings.warn("lanczos: poorly conditioned result (norm deviates from 1). "
                      "H may be non-Hermitian or k too small.")
    psi = psi * (1. / norm)

    return float(evals[0].real), psi


def lanczos_expm_multiply(
    apply, v0: "cytnx.UniTensor", dt: complex | float, k: int = 20
) -> "cytnx.UniTensor":
    """Compute exp(dt * H_eff) |v0> using a Lanczos-Krylov approximation.

    Builds a Krylov basis of size k and applies the matrix exponential only
    within that small subspace, then projects back to the original space.

    Parameters
    ----------
    apply : callable
        apply(v) -> v'  —  the matrix-vector product H|v>.
    v0 : UniTensor
        Initial vector.  Need not be normalised; the norm is preserved in
        the output (i.e. exp(dt*H) acts on the un-normalised v0).
    dt : complex | float
        Time argument.  Can be complex (e.g. dt = -1j*delta_t for real-time
        TDVP, or dt = -delta_tau for imaginary-time decay).
    k : int
        Maximum number of Lanczos iterations (Krylov dimension).

    Returns
    -------
    result : UniTensor   exp(dt * H_eff) |v0>,  same labels as v0.

    """
    norm0 = v0.Norm().item()
    T, vecs = _lanczos_iterations(apply, v0, k)

    # Initial state in Krylov basis: e0 = [1, 0, ..., 0]
    e0 = np.zeros(len(T))
    e0[0] = 1.0

    # Evolve in the small Krylov subspace and project back.
    evals, evecs = np.linalg.eigh(T)
    vt = evecs @ (np.exp(dt * evals) * (evecs.T @ e0))

    _labels = list(v0.labels())
    result = vt[0] * vecs[0]
    for i in range(1, len(vecs)):
        result = result + vt[i] * vecs[i]
        result.set_labels(_labels)

    # _lanczos_iterations normalises v0 to unit norm; restore original scale.
    result = result * norm0
    result.set_labels(_labels)
    return result


def _lanczos_iterations(apply, v0: "cytnx.UniTensor", k: int) -> tuple:
    """Core Lanczos iteration: build tridiagonal matrix T and Krylov basis.

    Returns
    -------
    T    : np.ndarray, shape (m, m)  tridiagonal Hermitian matrix (m <= k)
    vecs : list of UniTensor          orthonormal Krylov basis vectors

    Notes
    -----
    Cytnx's UniTensor subtraction (a - b) resets labels to ['0','1',...].
    We save the canonical labels from v0 and restore them after every
    subtraction so that apply() can always rely on named indices.
    """
    # Save labels once; used to restore after cytnx subtraction drops them.
    _labels = list(v0.labels())

    def _sub(a, b):
        """Subtract b from a and restore canonical labels."""
        result = a - b
        result.set_labels(_labels)
        return result

    # Normalise initial vector
    norm0 = v0.Norm().item()
    v = v0 * (1. / norm0)
    vecs = [v]

    T = np.zeros((k, k), dtype=float)

    # First Lanczos step
    #
    #  w = H|v_0> - alpha_0 |v_0>
    #
    w = apply(v)
    alpha = inner(v, w)                 # <v_0|H|v_0>
    T[0, 0] = float(alpha.real)
    w = _sub(w, alpha * v)

    for i in range(1, k):
        #
        #  beta_i = ||w||,  |v_i> = |w> / beta_i
        #
        beta = w.Norm().item()
        if beta < 1.e-13:
            # Krylov space is fully spanned — truncate T
            T = T[:i, :i]
            break

        v = w * (1. / beta)
        vecs.append(v)
        T[i - 1, i] = T[i, i - 1] = beta

        #
        #  w = H|v_i> - beta_i |v_{i-1}> - alpha_i |v_i>
        #  + full re-orthogonalisation against all previous basis vectors
        #    to suppress floating-point orthogonality loss.
        #
        w = _sub(apply(v), beta * vecs[-2])
        alpha = inner(v, w)             # <v_i|H|v_i>
        T[i, i] = float(alpha.real)
        w = _sub(w, alpha * v)

        # Re-orthogonalise w against all earlier basis vectors v_0 ... v_{i-1}.
        # Each inner product is cheap (element-wise) compared to apply(); total
        # overhead is O(k) inner products per step → negligible in practice.
        for prev_v in vecs[:-1]:
            w = _sub(w, inner(prev_v, w) * prev_v)

    else:
        T = T[:k, :k]

    return T, vecs
