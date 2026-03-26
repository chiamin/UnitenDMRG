"""Utility helpers for UniTensor-based MPS."""

from __future__ import annotations

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
        tmp = cytnx.UniTensor.zeros(tensor.shape())
        tmp.convert_from(tensor)
        tensor = tmp
    return tensor.get_block().numpy()


def print_bond(bond) -> None:
    """Debug print for one bond."""
    print(bond.type(), bond.qnums(), bond.getDegeneracies())


def print_bonds(tensor: "cytnx.UniTensor") -> None:
    """Debug print for all bonds and basic tensor metadata."""
    print(tensor.labels())
    print(tensor.shape())
    for bond in tensor.bonds():
        print_bond(bond)
