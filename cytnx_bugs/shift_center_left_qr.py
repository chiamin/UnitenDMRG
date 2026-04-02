"""Minimal reproducer: _shift_center_left_qr crashes for QN tensors.

Bug: QR (and SVD) on the tensor produced by the left-shift permutation
gives auxiliary bonds with wrong directions/qnums. The subsequent Contract
fails with "must have at least one ket-bond and one bra-bond".

Root cause: combination of qr-aux-qnums and svd-aux-qnums — the left-shift
permutation produces non-standard bond directions [i=OUT, r=IN | aux=?]
that trigger incorrect auxiliary qnums in both QR and SVD.
"""
import cytnx

# Build a minimal 3-site QN MPS with center at site 2 (rightmost).
bd1 = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0) >> 1])
bd_phys = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0) >> 1, cytnx.Qs(1) >> 1])

# Site 0: l(dim=1), i(dim=2), r(dim=2)
bd_r0 = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(0) >> 1, cytnx.Qs(1) >> 1])
A0 = cytnx.UniTensor([bd1, bd_phys, bd_r0], labels=["l", "i", "r"],
                      rowrank=2, dtype=cytnx.Type.Double)

# Site 1: l(dim=2), i(dim=2), r(dim=2)
bd_l1 = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0) >> 1, cytnx.Qs(1) >> 1])
bd_r1 = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(1) >> 1, cytnx.Qs(2) >> 1])
A1 = cytnx.UniTensor([bd_l1, bd_phys, bd_r1], labels=["l", "i", "r"],
                      rowrank=2, dtype=cytnx.Type.Double)

# Site 2: l(dim=2), i(dim=2), r(dim=1)
bd_l2 = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(1) >> 1, cytnx.Qs(2) >> 1])
bd_r2 = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(2) >> 1])
A2 = cytnx.UniTensor([bd_l2, bd_phys, bd_r2], labels=["l", "i", "r"],
                      rowrank=2, dtype=cytnx.Type.Double)

# Attempt left-shift: QR-decompose A2 with row_labels=["i", "r"]
# This is what _shift_center_left_qr does.
print("QR decomposition of site 2 with row_labels=['i', 'r']:")
try:
    Q, R = cytnx.linalg.Qr(A2)
    print("  QR succeeded")
    print("  Q labels:", Q.labels())
    print("  R labels:", R.labels())
    for label, bond in zip(Q.labels(), Q.bonds()):
        print(f"    Q/{label}: type={bond.type()}, qnums={bond.qnums()}")

    # Try to contract R with A1
    R.permute_(["l", "nr"])
    R.set_labels(["r", "nr"])
    left_new = cytnx.Contract(A1, R)
    print("  Contract(A1, R): OK")
except RuntimeError as e:
    first_line = str(e).split("\n")[2] if len(str(e).split("\n")) > 2 else str(e)
    print(f"  BUG: {first_line}")
