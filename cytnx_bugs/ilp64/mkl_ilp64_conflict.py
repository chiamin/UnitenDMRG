"""Minimal reproducer: numpy LP64 vs cytnx ILP64 MKL conflict.

cytnx calls set_mkl_ilp64() on import, forcing MKL into ILP64 mode.
numpy is compiled with LP64 (32-bit int LAPACK calls).
Whichever is imported first, the other breaks.

Case 1: import numpy first  → numpy OK, cytnx complex QN Contract returns 0
Case 2: import cytnx first  → cytnx OK, numpy eigh returns wrong results

Usage:
    conda run -n cytnx python scratch/mkl_ilp64_conflict.py
"""

# ---- Case 1: numpy first ------------------------------------------------
print("=== Case 1: import numpy before cytnx ===")

import subprocess, sys

# Switch the order of importing cytnx and numpy to see different errors
import numpy as np
import cytnx

# numpy eigh
np.random.seed(42)
a = np.random.randn(4,4); a = a + a.T
vals, vecs = np.linalg.eigh(a)
resid = max(np.linalg.norm(a @ vecs[:,i] - vals[i] * vecs[:,i]) for i in range(4))
print("numpy eigh correct:", resid < 1e-10)

# cytnx complex QN
sym = cytnx.Symmetry.U1()
phi = cytnx.UniTensor(
    [cytnx.Bond(cytnx.BD_IN, [[0],[1]], [1,1], [sym]),
     cytnx.Bond(cytnx.BD_OUT, [[0],[1]], [1,1], [sym])],
    rowrank=1, dtype=cytnx.Type.ComplexDouble)
phi.set_labels(["a","b"])
phi[0,0] = 1.0; phi[1,1] = 1j
result = complex(cytnx.Contract(phi.Dagger(), phi).item())
print("cytnx Contract correct:", abs(result - 2.0) < 1e-8)

