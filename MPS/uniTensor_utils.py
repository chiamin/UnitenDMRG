"""Utility helpers for UniTensor-based MPS."""

from __future__ import annotations

from itertools import product
from typing import Iterable

import cytnx
import numpy as np

def to_uniTensor(arr: np.ndarray) -> "cytnx.UniTensor":
    """Convert a numpy array to UniTensor."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(arr).__name__}.")
    return cytnx.UniTensor(cytnx.from_numpy(arr))


def to_numpy_array(tensor: "cytnx.UniTensor") -> np.ndarray:
    """Convert a UniTensor to numpy array (auto-convert blockform)."""
    if not isinstance(tensor, cytnx.UniTensor):
        raise TypeError(f"Expected cytnx.UniTensor, got {type(tensor).__name__}.")
    if tensor.is_blockform():
        return _blockform_to_numpy(tensor)
    return tensor.get_block().numpy()


def _blockform_to_numpy(tensor: "cytnx.UniTensor") -> np.ndarray:
    """Assemble a dense numpy array from a BlockUniTensor's blocks.

    Iterates over QN-conserving sector combinations, retrieves each block,
    and places it into the correct slice of the output array.
    """
    bonds = tensor.bonds()
    n_bonds = len(bonds)
    nsym = bonds[0].Nsym()

    # Per-bond: qnums, degeneracies, cumulative offsets, direction sign.
    bond_qns = []
    bond_offsets = []
    bond_signs = []
    for bond in bonds:
        bond_qns.append(bond.qnums())
        degs = bond.getDegeneracies()
        offsets = [0]
        for d in degs:
            offsets.append(offsets[-1] + d)
        bond_offsets.append(offsets)
        bond_signs.append(1 if bond.type() == cytnx.bondType.BD_KET else -1)

    is_complex = "Complex" in tensor.dtype_str()
    result = np.zeros(tensor.shape(), dtype=complex if is_complex else float)

    n_sectors = [len(qns) for qns in bond_qns]
    blk_idx = 0
    for combo in product(*[range(n) for n in n_sectors]):
        # Check QN conservation: sum of signed qnums == 0.
        conserved = True
        for s in range(nsym):
            if sum(bond_signs[b] * bond_qns[b][sec][s] for b, sec in enumerate(combo)) != 0:
                conserved = False
                break
        if not conserved:
            continue

        slices = tuple(
            slice(bond_offsets[b][sec], bond_offsets[b][sec + 1])
            for b, sec in enumerate(combo)
        )
        result[slices] = tensor.get_block(blk_idx).numpy()
        blk_idx += 1

    return result


def print_bond(bond) -> None:
    """Debug print for one bond."""
    print(bond.type(), bond.qnums(), bond.getDegeneracies())


def print_bonds(tensor: "cytnx.UniTensor") -> None:
    """Debug print for all bonds and basic tensor metadata."""
    print(tensor.labels())
    print(tensor.shape())
    for bond in tensor.bonds():
        print_bond(bond)


def is_complex_tensor(tensor: "cytnx.UniTensor") -> bool:
    """Return True if a UniTensor uses a complex dtype."""
    if not isinstance(tensor, cytnx.UniTensor):
        raise TypeError(f"Expected cytnx.UniTensor, got {type(tensor).__name__}.")
    return "Complex" in tensor.dtype_str()


def any_complex_tensors(tensors: Iterable["cytnx.UniTensor"]) -> bool:
    """Return True if any UniTensor in the iterable is complex."""
    for tensor in tensors:
        if is_complex_tensor(tensor):
            return True
    return False
