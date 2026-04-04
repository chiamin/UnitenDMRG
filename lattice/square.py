"""Square lattice with row-major 1D site mapping."""

from __future__ import annotations


class SquareLattice:
    """2D square lattice mapped to a 1D chain via row-major ordering.

    Site (x, y) maps to 1D index  y * Lx + x,  where
        x = 0, ..., Lx-1   (column)
        y = 0, ..., Ly-1   (row)

    Parameters
    ----------
    Lx, Ly : int
        Number of sites in the x and y directions.
    xpbc, ypbc : bool
        Periodic boundary conditions in x / y direction (default False = OBC).
    """

    def __init__(
        self, Lx: int, Ly: int, *, xpbc: bool = False, ypbc: bool = False,
    ) -> None:
        if Lx < 1 or Ly < 1:
            raise ValueError(f"Lx and Ly must be >= 1; got Lx={Lx}, Ly={Ly}.")
        self._Lx = Lx
        self._Ly = Ly
        self._xpbc = xpbc
        self._ypbc = ypbc
        self._bonds = self._build_bonds()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def Lx(self) -> int:
        return self._Lx

    @property
    def Ly(self) -> int:
        return self._Ly

    @property
    def N(self) -> int:
        """Total number of sites."""
        return self._Lx * self._Ly

    def idx(self, x: int, y: int) -> int:
        """Return the 1D index for site (x, y)."""
        return y * self._Lx + x

    def coord(self, i: int) -> tuple[int, int]:
        """Return (x, y) for 1D index i."""
        return i % self._Lx, i // self._Lx

    def bonds(self) -> list[tuple[int, int]]:
        """Return all nearest-neighbor bonds as (i, j) pairs with i < j."""
        return list(self._bonds)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @property
    def xpbc(self) -> bool:
        return self._xpbc

    @property
    def ypbc(self) -> bool:
        return self._ypbc

    def _build_bonds(self) -> list[tuple[int, int]]:
        bonds = []
        for y in range(self._Ly):
            for x in range(self._Lx):
                i = self.idx(x, y)
                # x-direction
                if x + 1 < self._Lx:
                    bonds.append((i, self.idx(x + 1, y)))
                elif self._xpbc and self._Lx > 1:
                    bonds.append((self.idx(0, y), i))
                # y-direction
                if y + 1 < self._Ly:
                    bonds.append((i, self.idx(x, y + 1)))
                elif self._ypbc and self._Ly > 1:
                    bonds.append((self.idx(x, 0), i))
        # Normalize: ensure i < j and remove duplicates
        bonds = list(set((min(a, b), max(a, b)) for a, b in bonds))
        bonds.sort()
        return bonds

    def __repr__(self) -> str:
        bc = []
        if self._xpbc:
            bc.append("xpbc")
        if self._ypbc:
            bc.append("ypbc")
        bc_str = f", bc={'+'.join(bc)}" if bc else ""
        return f"SquareLattice(Lx={self._Lx}, Ly={self._Ly}{bc_str})"
