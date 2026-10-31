"""Part 3.3 Kernels & k-NN: the notes' numbers, the kernel identities, and the visuals' datasets rebuilt from the same seeds."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import check, done  # noqa: E402
from common_rng import rng, gauss  # noqa: E402

check("sqrt(2.1) ~ 1.45", round(math.sqrt(2.1), 2), 1.45)
phi = lambda v: np.array([1, math.sqrt(2) * v[0], math.sqrt(2) * v[1], math.sqrt(2) * v[0] * v[1], v[0] ** 2, v[1] ** 2])
x, xp = np.array([1.0, 2]), np.array([3.0, 1])
check("x^T x' = 5, K = 36", [x @ xp, (1 + x @ xp) ** 2], [5, 36])
check("phi(x) = (1, sqrt2, 2sqrt2, 2sqrt2, 1, 4)", phi(x), [1, math.sqrt(2), 2 * math.sqrt(2), 2 * math.sqrt(2), 1, 4])
check("phi(x') = (1, 3sqrt2, sqrt2, 3sqrt2, 9, 1)", phi(xp), [1, 3 * math.sqrt(2), math.sqrt(2), 3 * math.sqrt(2), 9, 1])
check("terms 1 + 6 + 4 + 12 + 9 + 4 = 36", phi(x) * phi(xp), [1, 6, 4, 12, 9, 4])
r = np.random.default_rng(3)
for _ in range(5):
    a, b = r.normal(size=2), r.normal(size=2)
    check("(1 + a.b)^2 = phi(a).phi(b) for random a, b", (1 + a @ b) ** 2, phi(a) @ phi(b))
check("C(4,2) = 6; C(22,2) = 231; C(102,2) = 5151; C(103,3) = 176851", [math.comb(4, 2), math.comb(22, 2), math.comb(102, 2), math.comb(103, 3)], [6, 231, 5151, 176851])
check("gamma = 1: e^-1 ~ 0.37, e^-4 ~ 0.018", [round(math.exp(-1), 2), round(math.exp(-4), 3)], [0.37, 0.018])
a, b, g = r.normal(size=3), r.normal(size=3), 0.7
series = sum((2 * g * a @ b) ** k / math.factorial(k) for k in range(40))
check("RBF = e^{-g|x|^2} e^{-g|x'|^2} sum_k (2g x.x')^k / k!", math.exp(-g * np.sum((a - b) ** 2)), math.exp(-g * a @ a) * math.exp(-g * b @ b) * series)
# PSD argument: c^T K c = ||sum c_i phi(x_i)||^2
X = r.normal(size=(6, 2)); c = r.normal(size=6)
Kp = np.array([[(1 + u @ v) ** 2 for v in X] for u in X])
check("c^T K c = ||sum c_i phi(x_i)||^2 for the polynomial kernel", c @ Kp @ c, np.linalg.norm(sum(ci * phi(xi) for ci, xi in zip(c, X))) ** 2)
# gram visual default points: RBF / poly / laplacian PSD, sigmoid not
G = np.array([[-1.2, 0.4], [-0.3, 1.1], [0.6, 0.8], [1.3, -0.2], [0.2, -1.0], [-0.8, -0.7]])
k = {"linear": lambda u, v: u @ v, "poly": lambda u, v: (1 + u @ v) ** 2, "rbf": lambda u, v: math.exp(-np.sum((u - v) ** 2)),
     "laplacian": lambda u, v: math.exp(-np.linalg.norm(u - v)), "sigmoid": lambda u, v: math.tanh(u @ v - 1)}
mins = {n: np.linalg.eigvalsh(np.array([[f(u, v) for v in G] for u in G])).min() for n, f in k.items()}
print("   (gram visual smallest eigenvalues:", {n: round(v, 4) for n, v in mins.items()}, ")")
check("valid kernels give no negative eigenvalue on the default points", [mins[n] >= -1e-12 for n in ("linear", "poly", "rbf", "laplacian")], [1, 1, 1, 1])
check("the sigmoid kernel gives a negative eigenvalue there", mins["sigmoid"] < 0, 1)
# kernel ridge: (K + lam I) alpha = y solves the gradient condition
Kr = np.array([[math.exp(-0.5 * np.sum((u - v) ** 2)) for v in X] for u in X]); y = r.normal(size=6); lam = 0.3
al = np.linalg.solve(Kr + lam * np.eye(6), y)
check("gradient -2K(y - K alpha) + 2 lam K alpha = 0 at alpha-hat", -2 * Kr @ (y - Kr @ al) + 2 * lam * Kr @ al, np.zeros(6))
check("||beta||^2 = alpha^T K alpha (explicit polynomial map)", np.linalg.norm(np.array([phi(xi) for xi in X]).T @ c) ** 2, c @ Kp @ c)
# k-NN examples
check("mode([0,1,1,0,1]) = 1", np.bincount([0, 1, 1, 0, 1]).argmax(), 1)
w = 1 / np.array([1, 2, 4])
check("weighted vote: class 1 gets 1, class 0 gets 0.75", [w[0], w[1] + w[2]], [1, 0.75])
# kernel perceptron on the ring data with the degree-2 polynomial kernel converges (the circle is linear in phi-space)
rr = rng(81); P = []
for _ in range(20):
    t = rr() * 2 * math.pi; d = math.sqrt(rr()) * 0.95; P.append((d * math.cos(t), d * math.sin(t), -1))
for _ in range(28):
    t = rr() * 2 * math.pi; d = 1.9 + rr() * 0.6; P.append((d * math.cos(t), d * math.sin(t), 1))
P = np.array(P); alpha = np.zeros(len(P)); bb = 0.0
K2 = (1 + P[:, :2] @ P[:, :2].T) ** 2
for p in range(200):
    upd = 0
    for i in range(len(P)):
        if P[i, 2] * ((alpha * P[:, 2]) @ K2[:, i] + bb) <= 0:
            alpha[i] += 1; bb += P[i, 2]; upd += 1
    if not upd:
        break
print(f"   (kernel perceptron, poly d=2: converged after {p + 1} passes, {int(alpha.sum())} updates)")
check("kernel perceptron separates the ring data", upd, 0)
check("the plane x1^2 + x2^2 = 2.1 separates the ring data", np.all(np.sign(P[:, 0] ** 2 + P[:, 1] ** 2 - 2.1) == P[:, 2]), 1)
# knn regression visual: rebuild KREG, compute the validation curve
rr = rng(111); fx = lambda x: math.sin(1.3 * x) + 0.25 * x
def mk(n):
    out = []
    for _ in range(n):
        x = rr() * 6; out.append((x, fx(x) + 0.35 * gauss(rr)))
    return np.array(out)
tr, va = mk(60), mk(200)
def knn_reg(x, kk):
    idx = np.argsort(np.abs(tr[:, 0] - x), kind="stable")[:kk]
    return tr[idx, 1].mean()
curve = [np.mean([(knn_reg(x, kk) - y_) ** 2 for x, y_ in va]) for kk in range(1, 41)]
best = int(np.argmin(curve)) + 1
print(f"   (knn regression visual: best k = {best}, MSE at k=1 {curve[0]:.4f}, best {min(curve):.4f}, k=40 {curve[39]:.4f})")
check("validation MSE: k = 1 and k = 40 are both worse than the interior minimum", [curve[0] > min(curve), curve[39] > min(curve), 1 < best < 40], [1, 1, 1])
done()
