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


def product_state(config: list[int] | tuple[int, ...], phys_dim: int) -> MPS:
    """Create a basis-product-state MPS from per-site basis indices.

    Args:
        config: Basis index at each site, e.g. [0, 1, 0, ...].
        phys_dim: Uniform physical dimension used at every site.
    """
    if not config:
        raise ValueError("config must contain at least one site index.")
    if phys_dim <= 0:
        raise ValueError("phys_dim must be positive.")

    tensors = []
    for site, state in enumerate(config):
        if not isinstance(state, int):
            raise TypeError(
                f"config[{site}] must be int; got {type(state).__name__}."
            )
        if not 0 <= state < phys_dim:
            raise ValueError(
                f"config[{site}]={state} is outside [0, {phys_dim - 1}]."
            )
        arr = np.zeros((1, phys_dim, 1), dtype=float)
        arr[0, state, 0] = 1.0
        ut = cytnx.UniTensor(cytnx.from_numpy(arr), rowrank=2)
        ut.set_labels(["l", "i", "r"])
        tensors.append(ut)
    return MPS(tensors)


def product_state_qn(
    num_sites: int,
    physical_bond: "cytnx.Bond",
    physical_index: int,
) -> MPS:
    """Create a uniform QN product state using one basis index on all sites.

    This routine uses one-dimensional zero-charge virtual bonds on both sides.
    Therefore the chosen `physical_index` must belong to a physical sector whose
    qnums are all zeros; otherwise no compatible block exists and a `ValueError` is
    raised.
    """
    if num_sites <= 0:
        raise ValueError("num_sites must be positive.")
    if not isinstance(physical_index, int):
        raise TypeError(f"physical_index must be int; got {type(physical_index).__name__}.")
    if not isinstance(physical_bond, cytnx.Bond):
        raise TypeError(
            f"physical_bond must be cytnx.Bond; got {type(physical_bond).__name__}."
        )
    if physical_bond.Nsym() == 0:
        raise ValueError("physical_bond must carry quantum numbers (Nsym > 0).")

    phys_dim = physical_bond.dim()
    if not 0 <= physical_index < phys_dim:
        raise ValueError(
            f"physical_index={physical_index} is outside [0, {phys_dim - 1}]."
        )

    qnums = physical_bond.qnums()
    degeneracies = physical_bond.getDegeneracies()
    nsym = physical_bond.Nsym()
    zero_qnums = [0] * nsym

    sector = 0
    offset = physical_index
    while sector < len(degeneracies) and offset >= degeneracies[sector]:
        offset -= degeneracies[sector]
        sector += 1
    if sector >= len(degeneracies):
        raise ValueError("physical_index is inconsistent with physical_bond degeneracies.")
    if list(qnums[sector]) != zero_qnums:
        raise ValueError(
            "Chosen physical_index is not in the zero-qnum sector; "
            "cannot build product_state_qn with zero-charge virtual bonds."
        )

    syms = list(physical_bond.syms())
    b_left = cytnx.Bond(cytnx.BD_IN, [zero_qnums], [1], syms)
    b_phys = physical_bond.retype(cytnx.BD_IN)
    b_right = cytnx.Bond(cytnx.BD_OUT, [zero_qnums], [1], syms)

    tensors = []
    for _ in range(num_sites):
        ut = cytnx.UniTensor([b_left, b_phys, b_right], rowrank=2)
        ut.set_labels(["l", "i", "r"])
        blk = ut.get_block_(0)
        blk.fill(0)
        blk[0, offset, 0] = 1.0
        tensors.append(ut)
    return MPS(tensors)
