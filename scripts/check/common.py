"""Shared helpers for the per-part checks: check(name, got, want, tol) and done()."""
import sys

import numpy as np

_fail = []
_n = 0


def check(name, got, want, tol=1e-9):
    global _n
    _n += 1
    g, w = np.asarray(np.real_if_close(np.asarray(got, complex)), float), np.asarray(want, float)
    ok = g.shape == w.shape and np.allclose(g, w, atol=tol, rtol=0)
    if not ok:
        _fail.append(name)
        print(f"FAIL {name}: got {np.round(g, 6).tolist()} want {np.round(w, 6).tolist()}")
    return ok


def done():
    if _fail:
        print(f"{len(_fail)} of {_n} checks failed")
        sys.exit(1)
    print(f"all {_n} checks passed")
