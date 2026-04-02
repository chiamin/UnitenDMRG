"""Minimal reproducer: linalg.Svd produces wrong auxiliary bond qnums.

Bug: when row bond directions are mixed (not all the same), the auxiliary
bond on U and Vt gets incorrect qnums. cytnx uses a fixed formula that
ignores actual bond directions.
"""
import cytnx

# T: [a=OUT qn=[0,1], b=IN qn=[4] | c=OUT qn=[4]]
bd_a = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(0) >> 1, cytnx.Qs(1) >> 1])
bd_b = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(4) >> 1])
bd_c = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(4) >> 1])

T = cytnx.UniTensor([bd_a, bd_b, bd_c], labels=["a", "b", "c"], rowrank=2,
                     dtype=cytnx.Type.Double)

print("Input T bonds:")
for label, bond in zip(T.labels(), T.bonds()):
    print(f"  {label}: type={bond.type()}, qnums={bond.qnums()}")

U, S, Vt = cytnx.linalg.Svd(T)

print("\nU bonds:")
for label, bond in zip(U.labels(), U.bonds()):
    print(f"  {label}: type={bond.type()}, qnums={bond.qnums()}")

# BUG: U auxiliary bond (_aux_L) should have qnums [3, 4]
# (from conservation: (-1)*qa + (+1)*qb + (-1)*qaux == 0)
# but cytnx produces [-4, -3].
print("\nExpected auxiliary qnums: [[3], [4]]")
print("Conservation: (-1)*q_a + (+1)*q_b + (-1)*q_aux == 0")
print("  q_a=0, q_b=4 => q_aux=4;  q_a=1, q_b=4 => q_aux=3")
