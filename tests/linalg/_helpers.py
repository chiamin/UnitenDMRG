"""Helpers for linalg solver tests (dense and QN).

Dense path
----------
`vec` / `to_np` / `make_apply` wrap a numpy matrix M as an `apply(v) -> M @ v`
callable acting on rank-1 dense UniTensor vectors with label ['i'].  The
reference answer comes from numpy.

QN path
-------
A QN (BlockUniTensor) Krylov vector needs >= 2 legs to expose more than one
symmetry sector (a bare rank-1 QN tensor only stores a single block).  We model
the vector as an MPS-A-like rank-2 ket and the operator as a rank-4 tensor:

    |v>  : [l=BD_IN, i=BD_OUT]                         labels ['l','i']
    A    : [p=BD_IN, q=BD_OUT, l=BD_OUT, i=BD_IN]      labels ['p','q','l','i']

A's ket legs ('l','i') have the opposite direction to v so Contract pairs
IN<->OUT; its output legs use distinct labels ('p','q') so that A and
A.Dagger() contract cleanly when forming a Hermitian operator A^dag A.

`make_qn_hermitian_apply` returns a Hermitian (positive-definite if shift > 0)
operator built as  apply(v) = A^dag A v + shift v, entirely in cytnx — no hand
symmetrization of a rank-4 tensor.  `qn_dense_matrix` extracts the dense matrix
of any QN `apply` by probing it with canonical basis vectors, in the SAME global
ordering used by `qn_to_np`, so QN solver output can be compared against numpy
(eigh / solve / expm) without manual QN bookkeeping.
"""

from __future__ import annotations

import numpy as np
import cytnx


# ---------------------------------------------------------------------------
# Dense path
# ---------------------------------------------------------------------------

def vec(arr: np.ndarray) -> "cytnx.UniTensor":
    """Wrap a 1-D numpy array as a rank-1 UniTensor with label ['i']."""
    u = cytnx.UniTensor(cytnx.from_numpy(np.ascontiguousarray(arr)), rowrank=1)
    u.set_labels(["i"])
    return u


def to_np(v: "cytnx.UniTensor") -> np.ndarray:
    return v.get_block().numpy().ravel()


def make_apply(M: np.ndarray):
    """Return apply(v) = M @ v as a UniTensor->UniTensor callable.

    The result preserves the input dtype family (complex stays complex).
    """
    def apply(v: "cytnx.UniTensor") -> "cytnx.UniTensor":
        x = to_np(v)
        return vec(M @ x)
    return apply


# ---------------------------------------------------------------------------
# QN path
# ---------------------------------------------------------------------------

def _np_dtype(ut: "cytnx.UniTensor"):
    return complex if ut.dtype() == cytnx.Type.ComplexDouble else float


def make_qn_vector(l_sectors, l_degs, i_sectors, i_degs,
                   dtype: str = "real", seed: int = 0) -> "cytnx.UniTensor":
    """Random rank-2 U(1) QN ket [l=BD_IN, i=BD_OUT], labels ['l','i'].

    `dtype` is 'real' or 'complex'.  Each block is filled with random values.
    """
    rng = np.random.default_rng(seed)
    cplx = dtype == "complex"
    sym = cytnx.Symmetry.U1()
    ut_dtype = cytnx.Type.ComplexDouble if cplx else cytnx.Type.Double
    bl = cytnx.Bond(cytnx.BD_IN,  [[q] for q in l_sectors], l_degs, [sym])
    bi = cytnx.Bond(cytnx.BD_OUT, [[q] for q in i_sectors], i_degs, [sym])
    v = cytnx.UniTensor([bl, bi], labels=["l", "i"], dtype=ut_dtype)
    for b in range(v.Nblocks()):
        blk = v.get_block_(b)
        arr = rng.standard_normal(blk.shape())
        if cplx:
            arr = arr + 1j * rng.standard_normal(blk.shape())
        blk[:] = cytnx.from_numpy(np.ascontiguousarray(arr))
    return v


def _make_qn_A(l_sectors, l_degs, i_sectors, i_degs,
               dtype: str, seed: int) -> "cytnx.UniTensor":
    """Random QN operator A: ket legs ['l','i'] (reversed vs v), output legs
    ['p','q'] with distinct labels for clean A / A.Dagger() pairing."""
    rng = np.random.default_rng(seed)
    cplx = dtype == "complex"
    sym = cytnx.Symmetry.U1()
    ut_dtype = cytnx.Type.ComplexDouble if cplx else cytnx.Type.Double
    bl = cytnx.Bond(cytnx.BD_OUT, [[q] for q in l_sectors], l_degs, [sym])
    bi = cytnx.Bond(cytnx.BD_IN,  [[q] for q in i_sectors], i_degs, [sym])
    bp = cytnx.Bond(cytnx.BD_IN,  [[q] for q in l_sectors], l_degs, [sym])
    bq = cytnx.Bond(cytnx.BD_OUT, [[q] for q in i_sectors], i_degs, [sym])
    A = cytnx.UniTensor([bp, bq, bl, bi], labels=["p", "q", "l", "i"],
                        rowrank=2, dtype=ut_dtype)
    for b in range(A.Nblocks()):
        blk = A.get_block_(b)
        arr = rng.standard_normal(blk.shape())
        if cplx:
            arr = arr + 1j * rng.standard_normal(blk.shape())
        blk[:] = cytnx.from_numpy(np.ascontiguousarray(arr))
    return A


def make_qn_hermitian_apply(l_sectors, l_degs, i_sectors, i_degs,
                            dtype: str = "real", seed: int = 0,
                            shift: float = 1.0):
    """Hermitian QN operator apply(v) = A^dag A v + shift v.

    Positive-definite for shift > 0.  Returns (apply, A) where `apply` maps a
    rank-2 QN ket [l,i] to one with identical labels/directions.
    """
    A = _make_qn_A(l_sectors, l_degs, i_sectors, i_degs, dtype, seed)
    Ad = A.Dagger()                         # [p,q,l,i] with reversed directions
    _labels = ["l", "i"]

    def apply(v):
        Av = cytnx.Contract(A, v)           # v[l,i] -> Av[p,q]
        AdAv = cytnx.Contract(Ad, Av)       # Ad[p,q] · Av[p,q] -> [l,i]
        AdAv.set_labels(_labels)
        if shift != 0.0:
            r = AdAv + shift * v
            r.set_labels(_labels)
            return r
        return AdAv
    return apply, A


def make_qn_general_apply(l_sectors, l_degs, i_sectors, i_degs,
                          dtype: str = "real", seed: int = 0,
                          shift: float = 4.0):
    """General (non-Hermitian) well-conditioned QN operator apply(v) = A v + shift v.

    `shift` adds a multiple of the identity to keep the operator well away from
    singular, suitable for gmres tests.  Returns (apply, A).
    """
    A = _make_qn_A(l_sectors, l_degs, i_sectors, i_degs, dtype, seed)
    # A maps [l,i] -> [p,q]; to add shift*v and re-apply we need A to map
    # [l,i] -> [l,i].  Relabel A's output p->l, q->i so apply stays in [l,i].
    A_relabel = A.clone()
    A_relabel.set_labels(["l", "i", "_l", "_i"])  # p->l, q->i, l->_l, i->_i
    _labels = ["l", "i"]

    def apply(v):
        vv = v.clone()
        vv.set_labels(["_l", "_i"])
        Av = cytnx.Contract(A_relabel, vv)  # contract _l,_i -> output l,i
        Av.set_labels(_labels)
        r = Av + shift * v
        r.set_labels(_labels)
        return r
    return apply, A


def qn_to_np(v: "cytnx.UniTensor") -> np.ndarray:
    """Flatten a rank-2 QN vector to 1-D numpy: block index order, C-ravel."""
    return np.concatenate([v.get_block_(b).numpy().ravel()
                           for b in range(v.Nblocks())])


def qn_from_np(arr: np.ndarray, v_template: "cytnx.UniTensor") -> "cytnx.UniTensor":
    """Inverse of `qn_to_np`: scatter a flat numpy array back into a QN vector
    with the same block structure as `v_template` (block order, C-ravel)."""
    out = v_template.clone()
    off = 0
    for b in range(out.Nblocks()):
        blk = out.get_block_(b)
        shp = blk.shape()
        bsize = int(np.prod(shp))
        chunk = arr[off:off + bsize].reshape(shp)
        blk[:] = cytnx.from_numpy(np.ascontiguousarray(chunk.astype(_np_dtype(out))))
        off += bsize
    return out


def qn_dim(v: "cytnx.UniTensor") -> int:
    """Total stored dimension (sum of block sizes) of a QN vector."""
    return sum(int(np.prod(v.get_block_(b).shape()))
               for b in range(v.Nblocks()))


def qn_dense_matrix(apply, v_template: "cytnx.UniTensor") -> np.ndarray:
    """Dense matrix of a QN `apply`, in the same basis order as `qn_to_np`.

    Probes `apply` with each canonical basis vector e_g (one stored entry = 1,
    all others 0) and stacks the flattened outputs as columns.
    """
    D = qn_dim(v_template)
    cols = []
    for g in range(D):
        ej = v_template.clone()
        off = 0
        for b in range(ej.Nblocks()):
            blk = ej.get_block_(b)
            shp = blk.shape()
            bsize = int(np.prod(shp))
            z = np.zeros(shp, dtype=_np_dtype(ej))
            if off <= g < off + bsize:
                flat = z.ravel()
                flat[g - off] = 1.0
                z = flat.reshape(shp)
            blk[:] = cytnx.from_numpy(np.ascontiguousarray(z))
            off += bsize
        cols.append(qn_to_np(apply(ej)))
    return np.array(cols).T
