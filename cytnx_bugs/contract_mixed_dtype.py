"""Minimal reproducer: Contract(qn_real, qn_complex) fails.

Bug: cytnx.Contract(qn_real, qn_complex) raises
     "[ERROR] not support type with dtype=3".
     Affects QN tensors only; dense UniTensors can mix real and complex.
     The reverse order Contract(qn_complex, qn_real) does NOT fail.
"""
import cytnx

bd_in = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(0) >> 1, cytnx.Qs(1) >> 1])
bd_out = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(0) >> 1, cytnx.Qs(1) >> 1])

qn_real = cytnx.UniTensor([bd_in, bd_out], labels=["a", "b"], dtype=cytnx.Type.Double)
qn_complex = cytnx.UniTensor([bd_in, bd_out], labels=["b", "c"], dtype=cytnx.Type.ComplexDouble)

# Reverse order works
print("Contract(qn_complex, qn_real):")
try:
    result = cytnx.Contract(qn_complex, qn_real)
    print("  OK")
except RuntimeError as e:
    print("  FAILED:", e)

# This order fails
print("\nContract(qn_real, qn_complex):")
try:
    result = cytnx.Contract(qn_real, qn_complex)
    print("  OK")
except RuntimeError as e:
    print("  BUG:", e)
