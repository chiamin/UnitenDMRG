"""Heisenberg spin-1/2 chain MPO (test helper reference implementation)."""

from __future__ import annotations

import numpy as np

try:
    import cytnx
except ImportError as exc:
    raise ImportError("cytnx is required for tests/helpers/heisenberg.py.") from exc

from MPS.mpo import MPO


def _spin_operators() -> dict[str, np.ndarray]:
    """Return Sz, S+, S-, I as numpy arrays."""
    I  = np.eye(2, dtype=float)
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=float)
    Sp = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)
    Sm = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    return {"I": I, "Sz": Sz, "Sp": Sp, "Sm": Sm}


def _make_w(ops: dict, J: float, delta: float, h: float) -> np.ndarray:
    """Build the 5x2x2x5 bulk MPO tensor W with [l, ip, i, r] order."""
    d = 2
    D = 5
    I, Sz, Sp, Sm = ops["I"], ops["Sz"], ops["Sp"], ops["Sm"]

    W = np.zeros((D, d, d, D), dtype=float)

    W[0, :, :, 0] = I
    W[1, :, :, 0] = Sp
    W[2, :, :, 0] = Sm
    W[3, :, :, 0] = Sz
    W[4, :, :, 0] = h * Sz

    W[4, :, :, 1] = (J / 2.0) * Sm
    W[4, :, :, 2] = (J / 2.0) * Sp
    W[4, :, :, 3] = J * delta * Sz
    W[4, :, :, 4] = I
    return W


def _numpy_to_mpo_site(arr: np.ndarray) -> "cytnx.UniTensor":
    ut = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
    ut.set_labels(["l", "ip", "i", "r"])
    return ut


def heisenberg_mpo(
    N: int,
    J: float = 1.0,
    delta: float = 1.0,
    h: float = 0.0,
    dtype: type | np.dtype = float,
) -> MPO:
    """Build the spin-1/2 Heisenberg chain MPO (reference helper)."""
    if N < 2:
        raise ValueError("Need at least 2 sites for a spin chain.")

    out_dtype = np.dtype(dtype)
    ops = _spin_operators()
    W = _make_w(ops, J, delta, h)

    tensors = []
    for i in range(N):
        if i == 0:
            arr = W[4:5, :, :, :]
        elif i == N - 1:
            arr = W[:, :, :, 0:1]
        else:
            arr = W
        arr = np.array(arr, dtype=out_dtype, copy=True)
        tensors.append(_numpy_to_mpo_site(arr))
    return MPO(tensors)

