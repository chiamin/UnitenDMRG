"""Conjugate-gradient linear solver for Hermitian positive-definite A x = b."""

from __future__ import annotations

import warnings

import cytnx

from .inner import inner


def cg(
    apply,
    b: "cytnx.UniTensor",
    x0: "cytnx.UniTensor | None" = None,
    k: int = 200,
    tol: float = 1.e-10,
) -> tuple:
    """Solve A x = b with the conjugate-gradient method.

    A must be Hermitian positive-definite.  For general (non-Hermitian or
    indefinite) A, use `gmres` instead — CG will silently produce wrong
    answers on indefinite systems.

    Parameters
    ----------
    apply : callable
        apply(v) -> A|v>.  v and A|v> must share labels with b.
    b : UniTensor
        Right-hand side.
    x0 : UniTensor or None
        Initial guess.  If None, the zero vector is used.
    k : int
        Maximum number of CG iterations.
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

    def _add(a, c):
        result = a + c
        result.set_labels(_labels)
        return result

    bnorm = b.Norm().item()
    if bnorm < 1.e-15:
        x = b * 0.
        x.set_labels(_labels)
        return x, 0.

    # x0, r0 = b - A x0, p0 = r0
    if x0 is None:
        x = b * 0.
        x.set_labels(_labels)
        r = b.clone()
    else:
        x = x0.clone()
        r = _sub(b, apply(x))

    rnorm = r.Norm().item()
    if rnorm / bnorm < tol:
        return x, rnorm / bnorm

    p = r.clone()
    # rs_old = <r|r>.  For Hermitian PD A and real iterates rs_old is real;
    # we keep the imaginary part only as a sanity check.
    rs_old = inner(r, r)
    if abs(rs_old.imag) > 1.e-10 * abs(rs_old.real):
        warnings.warn(
            f"cg: <r|r> has non-negligible imaginary part ({rs_old.imag:.2e}); "
            "labels or Dagger may be wrong."
        )
    rs_old = float(rs_old.real)

    converged = False
    for i in range(k):
        Ap = apply(p)
        pAp = inner(p, Ap)
        if abs(pAp.imag) > 1.e-10 * max(abs(pAp.real), 1.):
            warnings.warn(
                f"cg: <p|A|p> has non-negligible imaginary part ({pAp.imag:.2e}); "
                "A may not be Hermitian."
            )
        pAp_real = float(pAp.real)
        if pAp_real <= 0.:
            warnings.warn(
                f"cg: encountered <p|A|p> = {pAp_real:.3e} <= 0; "
                "A is not positive definite — use gmres."
            )
            break

        alpha = rs_old / pAp_real

        # x <- x + alpha p
        x = _add(x, alpha * p)

        # r <- r - alpha A p
        r = _sub(r, alpha * Ap)

        rs_new_c = inner(r, r)
        rs_new = float(rs_new_c.real)
        rnorm = rs_new ** 0.5

        if rnorm / bnorm < tol:
            converged = True
            break

        beta = rs_new / rs_old
        # p <- r + beta p
        p = _add(r, beta * p)
        rs_old = rs_new

    # True residual (cheap: one extra apply).
    r_final = _sub(b, apply(x))
    res = r_final.Norm().item() / bnorm

    if not converged and res > max(tol * 10., 1.e-6):
        warnings.warn(
            f"cg: did not converge to tol={tol:.1e} in k={k} iterations "
            f"(final relative residual = {res:.3e})."
        )

    return x, res
