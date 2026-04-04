"""Davidson eigensolver for Hermitian operators."""

from __future__ import annotations

import numpy as np

import cytnx

from .inner import inner


def davidson(
    apply,
    v0: "cytnx.UniTensor",
    precond=None,
    k: int = 20,
    tol: float = 1.e-12,
) -> tuple:
    """Find the ground-state energy and eigenvector using the Davidson method.

    The Davidson algorithm expands the search subspace using a preconditioned
    residual, which can converge faster than Lanczos when the operator is
    close to diagonal-dominant (common for DMRG effective Hamiltonians).

    Parameters
    ----------
    apply : callable
        apply(v) -> v'  --  the matrix-vector product H|v>.
    v0 : UniTensor
        Initial vector (need not be normalised).
    precond : callable or None
        precond(r, theta) -> t  --  preconditioner that maps the residual
        `r` and current eigenvalue estimate `theta` to a correction vector.
        Typically the Jacobi (diagonal) preconditioner:
            t_i = r_i / (theta - H_ii)
        If None, the residual is used directly as the correction (equivalent
        to Lanczos-like expansion without preconditioning).
    k : int
        Maximum subspace dimension (number of expansion vectors).
    tol : float
        Convergence threshold on the residual norm ||H|psi> - E|psi>||.

    Returns
    -------
    E0 : float
        Ground-state energy (lowest eigenvalue).
    psi : UniTensor
        Ground-state vector, normalised to unit norm.
    """
    _labels = list(v0.labels())

    def _add(a, b):
        """Add two vectors and restore labels."""
        result = a + b
        result.set_labels(_labels)
        return result

    def _sub(a, b):
        """Subtract two vectors and restore labels."""
        result = a - b
        result.set_labels(_labels)
        return result

    # Normalise initial vector
    norm0 = v0.Norm().item()
    if norm0 < 1.e-15:
        raise ValueError("Initial vector v0 has near-zero norm.")
    v = v0 * (1. / norm0)

    # Subspace basis and H applied to each basis vector
    V = [v]
    HV = [apply(v)]

    for iteration in range(k):
        # Build subspace Hamiltonian  T[i,j] = <V[i]|HV[j]>
        # T is Hermitian: real-symmetric for real vectors, complex-Hermitian
        # for complex vectors.  np.linalg.eigh handles both.
        m = len(V)
        T = np.zeros((m, m), dtype=complex)
        for i in range(m):
            for j in range(i, m):
                val = complex(inner(V[i], HV[j]))
                T[i, j] = val
                T[j, i] = val.conjugate()

        # Solve small eigenproblem.
        # eigh returns real eigenvalues.  For real-symmetric T (stored as
        # complex), eigenvectors are real up to numerical noise; cast back
        # to real so that scalar * UniTensor preserves the tensor dtype.
        evals, evecs = np.linalg.eigh(T)
        theta = float(evals[0])
        c = evecs[:, 0]
        if np.max(np.abs(c.imag)) < 1.e-14:
            c = c.real

        # Build Ritz vector  psi = sum_i c[i] * V[i]
        psi = c[0] * V[0]
        for i in range(1, m):
            psi = _add(psi, c[i] * V[i])

        # Build  H|psi> = sum_i c[i] * HV[i]
        Hpsi = c[0] * HV[0]
        for i in range(1, m):
            Hpsi = _add(Hpsi, c[i] * HV[i])

        # Residual  r = H|psi> - theta * |psi>
        r = _sub(Hpsi, theta * psi)
        rnorm = r.Norm().item()

        if rnorm < tol:
            break

        # At max subspace size, return best result so far
        if iteration == k - 1:
            break

        # Correction vector
        if precond is not None:
            t = precond(r, theta)
            t.set_labels(_labels)
        else:
            t = r

        # Orthogonalise t against all existing basis vectors
        for vi in V:
            t = _sub(t, inner(vi, t) * vi)

        tnorm = t.Norm().item()
        if tnorm < 1.e-14:
            # Correction is in the span of V; cannot expand further
            break
        t = t * (1. / tnorm)

        V.append(t)
        HV.append(apply(t))

    # Final normalisation
    norm = psi.Norm().item()
    psi = psi * (1. / norm)
    psi.set_labels(_labels)

    return float(theta), psi
