"""Minimal reproducer: linalg.Qr produces wrong auxiliary bond qnums.

Bug: when row bond directions don't follow [IN, IN | OUT], the auxiliary
bond on Q and R gets incorrect qnums. Subsequent Contract can segfault.
"""
import cytnx

# T: [i=OUT, b=IN, m=OUT] with rowrank=2
bd_i = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(1) >> 1, cytnx.Qs(2) >> 1])
bd_b = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(1) >> 1, cytnx.Qs(2) >> 1])
bd_m = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(1) >> 1, cytnx.Qs(2) >> 1])

T = cytnx.UniTensor([bd_i, bd_b, bd_m], labels=["i", "b", "m"], rowrank=2,
                     dtype=cytnx.Type.Double)

print("Input T bonds:")
for label, bond in zip(T.labels(), T.bonds()):
    print(f"  {label}: type={bond.type()}, qnums={bond.qnums()}")

Q, R = cytnx.linalg.Qr(T)

print("\nQ bonds:")
for label, bond in zip(Q.labels(), Q.bonds()):
    print(f"  {label}: type={bond.type()}, qnums={bond.qnums()}")

print("\nR bonds:")
for label, bond in zip(R.labels(), R.bonds()):
    print(f"  {label}: type={bond.type()}, qnums={bond.qnums()}")

# BUG: R auxiliary bond has wrong qnums (e.g. [[-2],[-1]] instead of
# values that satisfy QN conservation with the m bond).
print("\nR has", R.Nblocks(), "blocks but QN conservation is violated.")
