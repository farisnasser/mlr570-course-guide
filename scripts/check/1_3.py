"""Part 1.3 Calculus & Optimization: the notes' derivatives, Jacobian, identities and the GD example, checked numerically."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import check, done  # noqa: E402

h = 1e-6
d = lambda f, x: (f(x + h) - f(x - h)) / (2 * h)
# 1.3.1
check("f = x^2: f'(3) = 6", d(lambda x: x * x, 3.0), 6, tol=1e-6)
check("linearization: 6(0.01) = 0.06; exact change 0.0601", [6 * 0.01, (3.01 ** 2 - 9)], [0.06, 0.0601], tol=1e-12)
check("secant slope of x^2 is 2x + h", (16 - 9) / 1, 2 * 3 + 1)
rules = [
    (lambda x: 99 + 0 * x, lambda x: 0), (lambda x: 3 * x, lambda x: 3), (lambda x: x ** 3, lambda x: 3 * x ** 2),
    (lambda x: x ** 2 + 3 * x, lambda x: 2 * x + 3), (lambda x: x ** 2 * x, lambda x: 3 * x ** 2), (lambda x: 1 / x, lambda x: -1 / x ** 2),
    (lambda x: math.sin(x ** 2), lambda x: 2 * x * math.cos(x ** 2)),
]
for i, (f, fp) in enumerate(rules):
    for x in (0.7, 1.9):
        check(f"derivative rule example {i + 1} at x={x}", d(f, x), fp(x), tol=1e-6)
# 1.3.2 partials of 3x^2 y
F = lambda x, y: 3 * x * x * y
for x, y in ((1, 0.5), (-0.8, 1.2)):
    check(f"df/dx = 6xy, df/dy = 3x^2 at ({x},{y})", [d(lambda t: F(t, y), x), d(lambda t: F(x, t), y)], [6 * x * y, 3 * x * x], tol=1e-6)
# 1.3.3 Jacobian of (3x^2 y, 2x + y^8)
G = lambda x, y: 2 * x + y ** 8
x, y = 1.0, 0.8
J = np.array([[d(lambda t: F(t, y), x), d(lambda t: F(x, t), y)], [d(lambda t: G(t, y), x), d(lambda t: G(x, t), y)]])
check("J = [[6xy, 3x^2], [2, 8y^7]] at (1, 0.8)", J, [[6 * x * y, 3 * x * x], [2, 8 * y ** 7]], tol=1e-6)
# linearization error shrinks quadratically with the square size (jacobian visual)
gaps = []
for e in (0.2, 0.1, 0.05):
    pts = [(x + e * a, y + e * b) for a in (-1, 1) for b in (-1, 1)]
    gaps.append(max(np.linalg.norm(np.array([F(*q), G(*q)]) - np.array([F(x, y), G(x, y)]) - J @ (np.array(q) - [x, y])) for q in pts))
check("jacobian gap ~ quarter when the square halves", [gaps[0] / gaps[1] > 3, gaps[1] / gaps[2] > 3.4], [1, 1])
# element-wise and reductions
w, xv = np.array([1.5, -2.0, 0.5]), np.array([0.3, 0.7, -1.2])
Jw = np.array([[(np.multiply(w + h * np.eye(3)[j], xv)[i] - np.multiply(w - h * np.eye(3)[j], xv)[i]) / (2 * h) for j in range(3)] for i in range(3)])
check("d(w o x)/dw = diag(x)", Jw, np.diag(xv), tol=1e-8)
check("d(w^T x)/dx = w^T", [(w @ (xv + h * e) - w @ (xv - h * e)) / (2 * h) for e in np.eye(3)], w, tol=1e-8)
# 1.3.4 chain / total derivative
for x in (-1.3, 0.4, 1.2):
    check(f"d sin(x^2)/dx = 2x cos(x^2) at {x}", d(lambda t: math.sin(t * t), x), 2 * x * math.cos(x * x), tol=1e-6)
    check(f"d(x + x^2)/dx = 1 + 2x at {x}", d(lambda t: t + t * t, x), 1 + 2 * x, tol=1e-6)
# neuron gradient: z = w^T x + b, a = sigma(z), L = a^2 (any L): grad_w L = dL/da * da/dz * x
sig = lambda z: 1 / (1 + math.exp(-z))
b = 0.3
Lw = lambda wv: sig(wv @ xv + b) ** 2
a = sig(w @ xv + b)
check("neuron: grad_w L = (dL/da)(da/dz) x", [(Lw(w + h * e) - Lw(w - h * e)) / (2 * h) for e in np.eye(3)], 2 * a * a * (1 - a) * xv, tol=1e-8)
# 1.3.5 identities
rng = np.random.default_rng(1)
A, bb, xx = rng.normal(size=(4, 3)), rng.normal(size=4), rng.normal(size=3)
num = lambda f, v: np.array([(f(v + h * e) - f(v - h * e)) / (2 * h) for e in np.eye(len(v))])
M = rng.normal(size=(3, 3))
check("grad x^T A x = (A + A^T) x", num(lambda v: v @ M @ v, xx), (M + M.T) @ xx, tol=1e-6)
check("grad ||Ax - b||^2 = 2 A^T (Ax - b)", num(lambda v: np.sum((A @ v - bb) ** 2), xx), 2 * A.T @ (A @ xx - bb), tol=1e-6)
av, bv, W = rng.normal(size=3), rng.normal(size=2), rng.normal(size=(3, 2))
gW = np.array([[((av @ (W + h * np.outer(np.eye(3)[i], np.eye(2)[j])) @ bv) - (av @ (W - h * np.outer(np.eye(3)[i], np.eye(2)[j])) @ bv)) / (2 * h) for j in range(2)] for i in range(3)])
check("grad_W a^T W b = a b^T", gW, np.outer(av, bv), tol=1e-6)
A1, B1, C1 = rng.normal(size=(2, 3)), rng.normal(size=(3, 4)), rng.normal(size=(4, 2))
check("tr(ABC) = tr(BCA) = tr(CAB)", [np.trace(A1 @ B1 @ C1), np.trace(B1 @ C1 @ A1)], [np.trace(C1 @ A1 @ B1)] * 2)
# 1.3.6 GD on w^T w
w0 = np.array([2.2, 1.4])
for eta in (0.05, 0.25, 0.75, 1.05):
    wt = w0.copy()
    for _ in range(8):
        wt = wt - eta * 2 * wt
    check(f"GD eta={eta}: 8 steps = (1-2 eta)^8 w0", wt, (1 - 2 * eta) ** 8 * w0)
check("0 < eta < 1 <=> |1 - 2 eta| < 1", [abs(1 - 2 * e) < 1 for e in (0.01, 0.5, 0.99, 1.01)], [1, 1, 1, 0])
# convexity visual: the non-convex curve has two minima (global and local)
g = np.linspace(-2.2, 2.2, 440001); fv = 0.25 * g ** 4 - g ** 2 + 0.3 * g
mins = [i for i in range(1, len(g) - 1) if fv[i] < fv[i - 1] and fv[i] < fv[i + 1]]
check("non-convex example: two local minima", len(mins), 2)
# hessian visual presets
for Hm, sign in (([[1.2, 0.4], [0.4, 0.8]], 1), ([[-1.2, 0.4], [0.4, -0.8]], -1)):
    ev = np.linalg.eigvalsh(np.array(Hm))
    check(f"preset {Hm}: eigenvalues share sign {sign}", np.all(np.sign(ev) == sign), 1)
check("saddle preset diag(1,-1) has mixed eigenvalues", sorted(np.linalg.eigvalsh(np.diag([1, -1]))), [-1, 1])
done()
