"""Check whether cytnx and numpy use LP64 or ILP64 BLAS integers.

LP64  = 32-bit int (4 bytes)
ILP64 = 64-bit int (8 bytes)

Usage:
    python cytnx_bugs/ilp64/check_blas_int_size.py
"""

print("=== cytnx ===")
try:
    import cytnx
    size = cytnx.__blasINTsize__
    mode = "ILP64" if size == 8 else "LP64"
    print(f"  __blasINTsize__ = {size}  ->  {mode}")
except ImportError:
    print("  cytnx not installed")

print()
print("=== numpy ===")
try:
    import numpy as np
    config = np.show_config(mode="dicts")
    blas_info = config.get("Build Dependencies", {}).get("blas", {})
    openblas_cfg = blas_info.get("openblas configuration", "")
    name = blas_info.get("name", "unknown")

    if "USE64BITINT" in openblas_cfg:
        print(f"  BLAS: {name}  ->  ILP64 (USE64BITINT)")
    elif "mkl" in name.lower():
        # MKL: check via mkl_service if available
        try:
            import mkl
            layer = mkl.get_version_string()
            print(f"  BLAS: MKL ({layer})")
            import os
            iface = os.environ.get("MKL_INTERFACE_LAYER", "not set")
            print(f"  MKL_INTERFACE_LAYER = {iface}")
        except ImportError:
            print(f"  BLAS: {name} (install mkl-service for details)")
    else:
        print(f"  BLAS: {name}, openblas config: {openblas_cfg or 'unknown'}")
        if openblas_cfg:
            print(f"  ->  LP64 (no USE64BITINT)")
        else:
            print(f"  ->  cannot determine LP64/ILP64 from config")
except ImportError:
    print("  numpy not installed")
except TypeError:
    # older numpy without mode="dicts"
    print("  numpy too old for show_config(mode='dicts'), printing raw config:")
    np.show_config()
