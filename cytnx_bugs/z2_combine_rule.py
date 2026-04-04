import cytnx

sym = cytnx.Symmetry.Zn(2)

# combine_rule can give 0, 1 or -1
print("=== Z2 combine_rule ===")
for a in range(-1, 3):
    for b in range(-1, 3):
        result = sym.combine_rule(a, b)
        print("  combine(", a, ",", b, ") =", result)

# However, Bond rejects qnums -1 ---
b_bad = cytnx.Bond(cytnx.BD_IN, [[-1], [0]], [1, 1], [sym]) # raise error
