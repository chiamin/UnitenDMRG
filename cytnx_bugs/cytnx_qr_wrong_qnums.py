"""cytnx bug: linalg.Qr produces wrong auxiliary bond qnums for BlockUniTensor
when row bonds are BD_OUT and col bond is BD_IN.

The auxiliary bond gets negative qnums that violate QN conservation:
no valid blocks can exist, yet cytnx stores nonzero blocks.
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


def check_conservation(t, name):
    """Check sum(IN qnums) == sum(OUT qnums) for all sector combinations."""
    labels = t.labels()
    in_bonds = [(lb, t.bond(lb)) for lb in labels if t.bond(lb).type() == cytnx.BD_IN]
    out_bonds = [(lb, t.bond(lb)) for lb in labels if t.bond(lb).type() == cytnx.BD_OUT]

    # For rank-2: just check pairwise
    if len(in_bonds) == 1 and len(out_bonds) == 1:
        in_lb, in_b = in_bonds[0]
        out_lb, out_b = out_bonds[0]
        n_valid = sum(1 for iq in in_b.qnums() for oq in out_b.qnums()
                      if iq[0] == oq[0])
        print("{}: {} valid sector pairs, Nblocks stored: {}".format(
            name, n_valid, t.Nblocks()))
        if t.Nblocks() > 0 and n_valid == 0:
            print("  → BUG: blocks stored but none satisfy QN conservation!")
        sys.stdout.flush()
    else:
        print("Good")


# T: row bonds are OUT, col bond is IN
T = cytnx.UniTensor([
    cytnx.Bond(cytnx.BD_OUT, [[0], [1]], [1, 1], [sym]),
    cytnx.Bond(cytnx.BD_IN,  [[2]],      [1],    [sym]),
    cytnx.Bond(cytnx.BD_OUT, [[1], [2]], [1, 1], [sym]),
], labels=["i", "b", "m"], rowrank=2)
cytnx.random.uniform_(T, -1.0, 1.0)

show("T", T)
check_conservation(T, "T")
print()

Q, R = cytnx.linalg.Qr(T)
Q.relabel_(Q.labels()[-1], "aux")
R.relabel_(R.labels()[0], "aux")

show("Q", Q)
print()
show("R", R)
print()
check_conservation(R, "R")
