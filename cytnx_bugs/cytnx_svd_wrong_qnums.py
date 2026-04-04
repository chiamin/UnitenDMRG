"""cytnx bug: linalg.Svd produces wrong auxiliary bond qnums for BlockUniTensor
when row bond directions do not follow the standard [IN, IN, ... | OUT, OUT, ...] pattern.

The auxiliary bond gets qnums computed as if all row bonds were IN, ignoring
their actual directions.  This produces sectors that violate QN conservation.
Subsequent Contract on these tensors can segfault.
"""
import sys
import cytnx

sym = cytnx.Symmetry.U1()


def show(name, t):
    print("{}:".format(name))
    for lb in t.labels():
        b = t.bond(lb)
        d = "IN" if b.type() == cytnx.BD_IN else "OUT"
        print("  {} = {} qn={}".format(lb, d, list(b.qnums())))
    sys.stdout.flush()


def check_aux_qnums(t, name):
    """Check whether the auxiliary bond qnums satisfy QN conservation.

    For a rank-3 tensor [a, b, aux] from SVD, conservation requires:
        sign(a)*qa + sign(b)*qb + sign(aux)*qaux == 0
    for every stored block.  We enumerate all possible (qa, qb) pairs and
    compute the expected qaux, then compare with the actual aux qnums.
    """
    from itertools import product

    labels = t.labels()
    bonds = [(lb, t.bond(lb)) for lb in labels]
    signs = [+1 if b.type() == cytnx.BD_IN else -1 for _, b in bonds]
    qn_lists = [list(b.qnums()) for _, b in bonds]

    row_signs = signs[:-1]
    row_qns = qn_lists[:-1]
    aux_sign = signs[-1]
    aux_qns_actual = set(q[0] for q in qn_lists[-1])

    expected = set()
    for combo in product(*row_qns):
        row_sum = sum(s * q[0] for s, q in zip(row_signs, combo))
        qaux = -row_sum // aux_sign
        expected.add(qaux)

    print("{}: aux bond qnums (actual) = {}".format(name, sorted(aux_qns_actual)))
    print("{}  aux bond qnums (expected from conservation) = {}".format(
        " " * len(name), sorted(expected)))
    if aux_qns_actual == expected:
        print("  → OK: QN conservation satisfied")
    else:
        print("  → BUG: aux qnums do not satisfy QN conservation!")
    sys.stdout.flush()


# Row bonds [OUT, IN] — non-standard directions.
T = cytnx.UniTensor([
    cytnx.Bond(cytnx.BD_OUT, [[0], [1]], [1, 1], [sym]),   # a
    cytnx.Bond(cytnx.BD_IN,  [[4]],      [1],    [sym]),   # b
    cytnx.Bond(cytnx.BD_OUT, [[4]],      [1],    [sym]),   # c
], labels=["a", "b", "c"], rowrank=2)
cytnx.random.uniform_(T, -1.0, 1.0)

show("T", T)
s, U, Vd = cytnx.linalg.Svd(T)
print()
show("U", U)
check_aux_qnums(U, "U")
print()

# Contract(U, s) segfaults because U has invalid QN sectors.
print("=== Contract(U, s): segfaults ===")
sys.stdout.flush()
try:
    res = cytnx.Contract(U, s)
    print("  OK (unexpected)")
except Exception as e:
    print("  FAILED: {}".format(e))
