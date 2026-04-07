"""GMRES linear solver for A x = b on UniTensor vectors."""

from __future__ import annotations

import warnings

import numpy as np

import cytnx

from .inner import inner


def gmres(
    apply,
    b: "cytnx.UniTensor",
    x0: "cytnx.UniTensor | None" = None,
    k: int = 30,
    tol: float = 1.e-10,
) -> tuple:
    """Solve A x = b with restart-free GMRES.

    Builds an Arnoldi-Krylov basis { A^i r0 : i < k } and solves the small
    least-squares problem min || beta e1 - H y || in that subspace, then
    projects back to the original space.  No restart: if k iterations are
    not enough, increase k or wrap this in an outer loop.

    A need not be Hermitian.  For Hermitian (positive-definite) A use `cg`,
    which is cheaper.

    Parameters
    ----------
    apply : callable
        apply(v) -> A|v>.  v and A|v> must share labels with b.
    b : UniTensor
        Right-hand side.
    x0 : UniTensor or None
        Initial guess.  If None, the zero vector (0 * b) is used.
    k : int
        Maximum number of Arnoldi iterations (Krylov dimension).
    tol : float
        Convergence tolerance on ||r|| / ||b||.

    Returns
    -------
    x : UniTensor
        Approximate solution, same labels as b.
    res : float
        Final relative residual ||b - A x|| / ||b||.
    """
    _labels = list(b.labels())

    def _sub(a, c):
        result = a - c
        result.set_labels(_labels)
        return result

    bnorm = b.Norm().item()
    if bnorm < 1.e-15:
        # b == 0  →  x = 0 is the exact solution.
        x = b * 0.
        x.set_labels(_labels)
        return x, 0.

    # r0 = b - A x0
    if x0 is None:
        x = b * 0.
        x.set_labels(_labels)
        r = b.clone()
    else:
        x = x0.clone()
        r = _sub(b, apply(x))

    beta = r.Norm().item()
    if beta / bnorm < tol:
        return x, beta / bnorm

    # Arnoldi basis and (k+1) x k Hessenberg matrix
    V = [r * (1. / beta)]
    H = np.zeros((k + 1, k), dtype=complex)

    converged_at = k
    for j in range(k):
        #
        #  w = A |v_j>;  modified Gram-Schmidt against v_0 ... v_j
        #
        w = apply(V[j])
        for i in range(j + 1):
            hij = inner(V[i], w)         # <v_i|w>
            H[i, j] = hij
            w = _sub(w, hij * V[i])

        h_next = w.Norm().item()
        H[j + 1, j] = h_next

        if h_next < 1.e-13:
            # Krylov space exhausted; exact solution lies in span(V).
            converged_at = j + 1
            break

        V.append(w * (1. / h_next))

        # Cheap convergence check: solve the (j+2) x (j+1) least-squares
        # problem and read off the residual norm = |last component|.
        #
        # We do this every step because each apply() is the expensive part;
        # the small lstsq is negligible.
        e1 = np.zeros(j + 2, dtype=complex)
        e1[0] = beta
        y, *_ = np.linalg.lstsq(H[: j + 2, : j + 1], e1, rcond=None)
        res_est = np.linalg.norm(H[: j + 2, : j + 1] @ y - e1)
        if res_est / bnorm < tol:
            converged_at = j + 1
            break
    else:
        converged_at = k

    # Final small least-squares solve in the converged subspace.
    m = converged_at
    e1 = np.zeros(m + 1, dtype=complex)
    e1[0] = beta
    y, *_ = np.linalg.lstsq(H[: m + 1, : m], e1, rcond=None)

    # x = x0 + V[:m] @ y    (UniTensor addition drops labels — restore)
    for i in range(m):
        x = x + complex(y[i]) * V[i]
        x.set_labels(_labels)

    # True residual (cheap: one extra apply).
    r_final = _sub(b, apply(x))
    res = r_final.Norm().item() / bnorm

    if res > max(tol * 10., 1.e-6):
        warnings.warn(
            f"gmres: did not converge to tol={tol:.1e} in k={k} iterations "
            f"(final relative residual = {res:.3e}). Increase k or restart."
        )

    return x, res
