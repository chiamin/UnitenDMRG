"""Minimal reproducer: arithmetic resets labels.

Bug: a + b and a - b reset bond labels to ["0", "1", ...].
Scalar multiplication a * scalar is NOT affected.
"""
import cytnx

a = cytnx.UniTensor(cytnx.zeros([2, 3]), rowrank=1)
a.set_labels(["x", "y"])

b = cytnx.UniTensor(cytnx.zeros([2, 3]), rowrank=1)
b.set_labels(["x", "y"])

print("Before arithmetic:")
print("  a.labels():", a.labels())  # ['x', 'y']
print("  b.labels():", b.labels())  # ['x', 'y']

c = a + b
print("\nAfter a + b:")
print("  c.labels():", c.labels())  # BUG: ['0', '1'] instead of ['x', 'y']

d = a - b
print("\nAfter a - b:")
print("  d.labels():", d.labels())  # BUG: ['0', '1'] instead of ['x', 'y']

e = a * 2.0
print("\nAfter a * 2.0:")
print("  e.labels():", e.labels())  # OK: ['x', 'y']
