"""Shared helpers for MPO tests: full dense matrix extraction."""

import numpy as np


def _get_dense_numpy(ut):
    """Return the full dense numpy array for a UniTensor (dense or QN)."""
    if ut.is_blockform():
        from itertools import product as iproduct
        shape = list(ut.shape())
        arr = np.zeros(shape, dtype=ut.get_block().numpy().dtype)
        for idx in iproduct(*[range(s) for s in shape]):
            try:
                arr[idx] = ut.at(list(idx)).value
            except (RuntimeError, ValueError):
                continue
        return arr
    return ut.get_block().numpy()


def mpo_full_matrix(mpo):
    """Return the d^N x d^N dense matrix by pure numpy contraction.

    W[0]["l"] and W[-1]["r"] each have dim=1, so the boundary condition is
    simply vec = [1.0] on the left and val = vec[0] on the right.
    """
    N = len(mpo)
    d = mpo.phys_dims[0]

    W_arrs = []
    for p in range(N):
        w_perm = mpo[p].permute(["l", "ip", "i", "r"])
        W_arrs.append(_get_dense_numpy(w_perm))

    from itertools import product as iproduct

    out_dtype = np.result_type(*[arr.dtype for arr in W_arrs])
    total_dim = d**N
    mat = np.zeros((total_dim, total_dim), dtype=out_dtype)

    for bra_idx in iproduct(range(d), repeat=N):
        for ket_idx in iproduct(range(d), repeat=N):
            vec = np.array([1.0], dtype=out_dtype)
            for p in range(N):
                M = W_arrs[p][:, bra_idx[p], ket_idx[p], :]
                vec = M.T @ vec
            val = vec[0]
            bra_flat = sum(bra_idx[p] * d**(N - 1 - p) for p in range(N))
            ket_flat = sum(ket_idx[p] * d**(N - 1 - p) for p in range(N))
            mat[bra_flat, ket_flat] = val

    return mat
