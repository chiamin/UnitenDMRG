# TODO

## Remove cytnx arithmetic label workaround

**Condition:** Wait until cytnx fixes the bug where `a + b` and `a - b` reset
labels to `['0', '1', ...]` instead of preserving them. Scalar multiplication
`a * scalar` is not affected.

**Locations to clean up once the bug is fixed:**

- `MPS/linalg.py` — `_sub()` helper and its `result.set_labels(_labels)` call
- `MPS/linalg.py` — `psi.set_labels(_labels)` inside the `lanczos()` reconstruction loop
- `DMRG/effective_operators.py` — `result.set_labels(_labels)` inside `EffOperator.apply()` after adding rank-1 terms
