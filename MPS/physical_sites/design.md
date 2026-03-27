# PhysicalSite Design

## Overview

A `PhysicalSite` encapsulates the local Hilbert space for one lattice site:
the physical bond, a set of named operators, and a set of named single-site
states.  A single class is used for all site types; variation lives in the
data, not the class hierarchy.  Factory functions (`spin_half`, `electron`,
…) construct the appropriate `PhysicalSite` instance.

---

## Class interface: `PhysicalSite`

```python
class PhysicalSite:
    type_name: str          # e.g. "SpinHalf", "Electron"

    @property
    def bond(self) -> cytnx.Bond:
        """Physical bond (BD_IN, pointing toward site 0)."""

    @property
    def dim(self) -> int:
        """Physical Hilbert-space dimension."""

    def qns(self) -> list[str]:
        """List of QN labels, e.g. ["Sz"] or ["N","Sz"]. Empty = no QN."""

    def op(self, name: str) -> cytnx.UniTensor:
        """Return operator tensor with labels ["ip", "i"] (bra, ket)."""

    def state(self, name: str) -> cytnx.UniTensor:
        """Return rank-1 state tensor with label ["i"]."""

    def available_ops(self) -> list[str]:
        """Names of all registered operators."""

    def available_states(self) -> list[str]:
        """Names of all registered states."""
```

`op()` and `state()` raise `KeyError` for unknown names.

---

## `spin_half(qn=None)` factory

### Physical basis (fixed)

| index | state  | Sz    |
|-------|--------|-------|
| 0     | \|dn⟩  | −1/2  |
| 1     | \|up⟩  | +1/2  |

### `qn=None` — dense tensors

- `bond = cytnx.Bond(cytnx.BD_IN, 2)`  (no symmetry)
- All operators are dense rank-2 UniTensors with labels `["ip", "i"]`.
- All states are dense rank-1 UniTensors with label `["i"]`.
- Dtype: real (`float64`) for I, Sz, Sp, Sm, Sx; complex (`complex128`) for Sy.

Operators:

| name | matrix (row=ip, col=i) |
|------|------------------------|
| I    | [[1,0],[0,1]]          |
| Sz   | [[-0.5,0],[0,0.5]]     |
| Sp   | [[0,0],[1,0]]  (dn→up, index [1,0]) |
| Sm   | [[0,1],[0,0]]  (up→dn, index [0,1]) |
| Sx   | [[0,0.5],[0.5,0]]      |
| Sy   | [[0,-0.5j],[0.5j,0]]   |

States: `dn` = [1,0], `up` = [0,1].

### `qn="Sz"` — U(1) Sz block-sparse tensors

Bond construction:

```python
sym  = cytnx.Symmetry.U1()
# sector 0: qn = -1 (|dn>)
# sector 1: qn = +1 (|up>)
bond = cytnx.Bond(cytnx.BD_IN, [[-1], [1]], [1, 1], [sym])
bra  = bond.redirect()   # BD_OUT
```

States (rank-1, label `["i"]`):

```python
s_dn = cytnx.UniTensor([bond], rowrank=1)
s_dn.set_labels(["i"])
s_dn.get_block_(0)[0] = 1.0   # sector 0 (qn=-1) → |dn>

s_up = cytnx.UniTensor([bond], rowrank=1)
s_up.set_labels(["i"])
s_up.get_block_(1)[0] = 1.0   # sector 1 (qn=+1) → |up>
```

Operators (labels `["ip", "i"]`, rowrank=1):

- **I, Sz** — Sz-diagonal → block-sparse with bonds `[bra, bond]`.
  Set blocks directly with `get_block_(sector_idx)`.

  ```python
  I_op  = cytnx.UniTensor([bra, bond], rowrank=1)
  I_op.get_block_(0)[0,0]  = 1.0
  I_op.get_block_(1)[0,0]  = 1.0

  Sz_op = cytnx.UniTensor([bra, bond], rowrank=1)
  Sz_op.get_block_(0)[0,0] = -0.5   # dn: Sz = -1/2
  Sz_op.get_block_(1)[0,0] =  0.5   # up: Sz = +1/2
  ```

- **Sp, Sm** — Sz changes by ±1 → dense 2×2 tensors (non-zero Sz flux).

  ```python
  Sp_op = cytnx.UniTensor(cytnx.zeros([2,2]), rowrank=1)
  Sp_op.set_labels(["ip","i"])
  Sp_op[1,0] = 1.0   # |dn> → |up>:  Sp[up(1), dn(0)] = 1

  Sm_op = cytnx.UniTensor(cytnx.zeros([2,2]), rowrank=1)
  Sm_op.set_labels(["ip","i"])
  Sm_op[0,1] = 1.0   # |up> → |dn>:  Sm[dn(0), up(1)] = 1
  ```

- **Sx, Sy** — not defined under U(1) Sz; `op("Sx")` / `op("Sy")` raise `KeyError`.

---

## Design decisions

1. **One class, factory functions** — `PhysicalSite` holds data; factories
   (`spin_half`, `electron`, …) build and populate it.  No subclassing needed
   because the interface is uniform.

2. **Direct element assignment** — operator and state tensors are built by
   directly setting UniTensor block elements (`get_block_(idx)[...]`), avoiding
   numpy round-trips for QN tensors.

3. **Real dtype by default** — use `float64` wherever the matrix is real;
   only Sy uses `complex128`.

4. **Sp/Sm as dense when QN active** — S± carry non-zero Sz flux (ΔSz = ±1),
   so they cannot live in the QN-symmetric block structure of the physical
   bond alone.  They are stored as plain dense 2×2 tensors.

5. **Physical basis is fixed** — index 0 = |dn⟩ (Sz=−1/2), index 1 = |up⟩
   (Sz=+1/2).  This is documented once here and relied upon everywhere.

6. **`type_name` field** — stored on the instance so that code working with
   mixed site types can identify what it has without isinstance checks.

7. **`qns()` returns a list** — empty list means no QN; `["Sz"]` means U(1) Sz;
   `["N","Sz"]` means two conserved quantities.  Replaces a boolean `has_qn`.

---

## Open question

**State bond direction for product-state construction.**

`state()` currently returns a rank-1 UniTensor with only the physical index
`["i"]` (dimension = site dimension).  When constructing a product-state MPS,
the virtual bonds (dimension 1) must be attached.  Two options:

- **Option A** — `state()` returns rank-1; the MPS builder appends dim-1
  virtual bonds when assembling the product state.
- **Option B** — `state()` returns rank-3 `["l","i","r"]` with dim-1 virtual
  bonds already attached, bond direction `BD_IN` pointing left (toward site 0).

The user's note "1-site state tensor 的 bond(dim=1) 往左" suggests Option B,
but needs confirmation before implementation.

---

## Files to create

```
MPS/physical_index/
    __init__.py
    site.py        # PhysicalSite class
    spin_half.py   # spin_half() factory
```

Custom site types: define the physical `bond`, build `ops` and `states` dicts
of UniTensors following the conventions above, then call
`PhysicalSite(bond, ops, states, type_name, qn_labels)`.
