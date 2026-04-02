"""Minimal reproducer: convert_from and to_dense crash for QN → dense.

Bug: UniTensor.convert_from(block_tensor) and block_tensor.to_dense() both
raise "[ERROR] fatal internal, cannot call on a un-initialize UniTensor_base"
when converting BlockUniTensor (QN) to dense. The reverse direction
(dense → QN) works fine.

NOTE: This bug only affects the LP64 build at ~/.local/cytnx.
The conda cytnx package does NOT have this bug.
"""
import cytnx

bd_in = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(1) >> 1, cytnx.Qs(0) >> 1])
bd_out = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(1) >> 1, cytnx.Qs(0) >> 1])

t_qn = cytnx.UniTensor([bd_in, bd_out], labels=["a", "b"],
                        dtype=cytnx.Type.Double)

blk0 = cytnx.zeros([1, 1], dtype=cytnx.Type.Double)
blk0[0, 0] = 3.14
t_qn.put_block(blk0, qidx=[0, 0])

blk1 = cytnx.zeros([1, 1], dtype=cytnx.Type.Double)
blk1[0, 0] = 2.71
t_qn.put_block(blk1, qidx=[1, 1])

# Test convert_from (QN → dense)
print("convert_from (QN → dense):")
tmp = cytnx.UniTensor.zeros(t_qn.shape(), dtype=cytnx.Type.Double)
try:
    tmp.convert_from(t_qn)
    print("  OK:", tmp.get_block().numpy())
except RuntimeError as e:
    print("  BUG:", str(e).split("\n")[2])

# Test to_dense
print("\nto_dense:")
try:
    dense = t_qn.to_dense()
    print("  OK:", dense.get_block().numpy())
except RuntimeError as e:
    print("  BUG:", str(e).split("\n")[2])

# Reverse direction works
print("\nconvert_from (dense → QN):")
t_qn2 = cytnx.UniTensor([bd_in, bd_out], labels=["a", "b"],
                         dtype=cytnx.Type.Double)
import numpy as np
ut_dense = cytnx.UniTensor(cytnx.from_numpy(np.diag([3.14, 2.71])))
ut_dense.set_labels(["a", "b"])
try:
    t_qn2.convert_from(ut_dense, True)
    print("  OK")
    for i in range(t_qn2.Nblocks()):
        print(f"    block {i}:", t_qn2.get_block(i).numpy())
except RuntimeError as e:
    print("  FAILED:", str(e).split("\n")[2])
