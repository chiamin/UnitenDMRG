"""Initialization helpers for building UniTensor-based MPS objects."""

from __future__ import annotations

import numpy as np

import cytnx

from .mps import MPS


def random_mps(
    num_sites: int,
    phys_dim: int,
    bond_dim: int,
    *,
    dtype: np.dtype | type = float,
    seed: int | None = None,
    normalize: bool = True,
) -> MPS:
    """Create a random open-boundary MPS with uniform physical and bond dimensions."""
    if num_sites <= 0:
        raise ValueError("num_sites must be positive.")
    if phys_dim <= 0:
        raise ValueError("phys_dim must be positive.")
    if bond_dim <= 0:
        raise ValueError("bond_dim must be positive.")

    rng = np.random.default_rng(seed)
    bond_dims = [1, 1] if num_sites == 1 else [1] + [bond_dim] * (num_sites - 1) + [1]

    tensors = []
    out_dtype = np.dtype(dtype)
    for site in range(num_sites):
        shape = (bond_dims[site], phys_dim, bond_dims[site + 1])
        if np.issubdtype(out_dtype, np.complexfloating):
            arr = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
            arr = arr.astype(out_dtype, copy=False)
        else:
            arr = rng.standard_normal(shape).astype(out_dtype, copy=False)

        ut = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
        ut.set_labels(["l", "i", "r"])
        tensors.append(ut)

    mps = MPS(tensors)
    if normalize:
        mps.orthogonalize()
        mps.normalize()
    return mps
