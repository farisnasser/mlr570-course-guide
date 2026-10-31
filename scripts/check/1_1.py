"""Part 1.1 Linear Algebra: every number in the notes' worked examples and every number the visuals open on.

Notes numbers are checked to the precision the notes print; visual defaults to float precision. The JavaScript side
(src/viz/la.js, parts/1.1/viz.js) is compared against the same values in the browser audit (see the add-notes skill).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import check, done  # noqa: E402

det, inv, rank = np.linalg.det, np.linalg.inv, np.linalg.matrix_rank

# ---- 1.1.1 inverse example
A = np.array([[4, 7], [2, 6]])
check("det [[4,7],[2,6]] = 24-14 = 10", det(A), 10)
check("A^-1 = 1/10 [[6,-7],[-2,4]]", inv(A), np.array([[6, -7], [-2, 4]]) / 10)
check("A A^-1 = 1/10 [[10,0],[0,10]] = I", A @ (np.array([[6, -7], [-2, 4]]) / 10), np.eye(2))
check("det [[2,4],[1,2]] = 0", det(np.array([[2, 4], [1, 2]])), 0)
# transpose example and rules
At = np.array([[1, 2, 3], [4, 5, 6]]).T
check("A^T of [[1,2,3],[4,5,6]]", At, [[1, 4], [2, 5], [3, 6]])
rng = np.random.default_rng(0)
B1, B2 = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
check("(AB)^T = B^T A^T", (B1 @ B2).T, B2.T @ B1.T)
check("(AB)^-1 = B^-1 A^-1", inv(B1 @ B2), inv(B2) @ inv(B1))
check("(A^T)^-1 = (A^-1)^T", inv(B1.T), inv(B1).T)

# ---- 1.1.2 rank example L, three cases
Lm = np.array([[1, 0, 2, 3, 4], [0.5, 0, 1, 1.5, 2], [2, 0, 4, 6, 8], [1, 0, 2, 3, 4], [2, 0, 4, 6, 8]])
check("rank(L) = 1", rank(Lm), 1)
check("R2 = R1/2, R3 = 2R1, R4 = R1, R5 = 2R1", [Lm[1], Lm[2], Lm[3], Lm[4]], [Lm[0] / 2, 2 * Lm[0], Lm[0], 2 * Lm[0]])
A1, b1 = np.array([[1, 1], [2, -1]]), np.array([3, 0])
check("case 1: x = 1, y = 2", np.linalg.solve(A1, b1), [1, 2])
A2, b2 = np.array([[1, 1], [2, 2]]), np.array([2, 5])
check("case 2: rank 1 and b not in Col(A)", [rank(A2), rank(np.c_[A2, b2])], [1, 2])
check("case 2: R2 - 2R1 gives 0 = 1", (b2[1] - 2 * b2[0]), 1)
A3, b3 = np.array([[1, 2], [2, 4]]), np.array([4, 8])
check("case 3: rank(A) = rank([A|b]) = 1", [rank(A3), rank(np.c_[A3, b3])], [1, 1])
for t in (-1.0, 0.0, 2.5):
    check(f"case 3: x = (4-2t, t) solves, t={t}", A3 @ np.array([4 - 2 * t, t]), b3)
# the solve visual: b dragged onto Col(A) in case 2 gives (k - t, t)
check("case 2 with b on the line: (k - t, t)", A2 @ np.array([3 - 1.0, 1.0]), [3, 6])

# ---- 1.1.3 null space examples
An = np.array([[1, 2], [2, 4]])
check("A (-2, 1) = 0", An @ np.array([-2, 1]), [0, 0])
check("rank 1 + nullity 1 = 2", rank(An) + (2 - rank(An)), 2)
A3d = np.array([[1, 2, -1], [2, 4, -2]])
check("rank of the 2x3 example = 1", rank(A3d), 1)
check("(1,0,1) and (0,1,2) are erased", [A3d @ [1, 0, 1], A3d @ [0, 1, 2]], [[0, 0], [0, 0]])
check("they are independent (span a plane)", rank(np.array([[1, 0, 1], [0, 1, 2]])), 2)
# nullspace3d visual defaults: x = (1,0,0), z = 0.8(1,0,1) + 0.6(0,1,2)
x, z = np.array([1, 0, 0]), 0.8 * np.array([1, 0, 1]) + 0.6 * np.array([0, 1, 2])
check("visual: x + z = (1.8, 0.6, 2)", x + z, [1.8, 0.6, 2.0])
check("visual: A(x+z) = Ax = (1, 2)", [A3d @ (x + z), A3d @ x], [[1, 2], [1, 2]])

# ---- 1.1.4 diagonal, symmetric, orthogonal, rank-one
D3 = np.diag([3, 7, 2])
check("Dx = (3x1, 7x2, 2x3)", D3 @ np.array([1.0, 1.0, 1.0]), [3, 7, 2])
S = np.array([[2, -1, 0], [-1, 3, 4], [0, 4, 5]])
w, Qs = np.linalg.eigh(S)
check("symmetric: Q Lambda Q^T = A with orthonormal Q", [Qs @ np.diag(w) @ Qs.T, Qs.T @ Qs], [S, np.eye(3)])
Q = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
check("Q^T Q = I", Q.T @ Q, np.eye(2))
check("Q is a 45 degree rotation (det 1, angle of Qe1 = -45 deg)", [det(Q), np.degrees(np.arctan2(Q[1, 0], Q[0, 0]))], [1, -45])
u, v = np.array([1, 2]), np.array([3, 4, 5])
check("u v^T = [[3,4,5],[6,8,10]]", np.outer(u, v), [[3, 4, 5], [6, 8, 10]])
check("rank(u v^T) = 1", rank(np.outer(u, v)), 1)

# ---- 1.1.5 / 1.1.6 eigen + diagonalization
A = np.array([[2, 1], [1, 2]])
check("(2-l)^2 - 1 = 0 at l = 3, 1", [(2 - 3) ** 2 - 1, (2 - 1) ** 2 - 1], [0, 0])
check("eigenvalues of [[2,1],[1,2]] = 3, 1", sorted(np.linalg.eigvals(A).real, reverse=True), [3, 1])
check("A v1 = 3 v1, A v2 = 1 v2", [A @ [1, 1], A @ [1, -1]], [[3, 3], [1, -1]])
P, Dm = np.array([[1, 1], [1, -1]]), np.diag([3, 1])
check("P^-1 = 1/(-2) [[-1,-1],[-1,1]] = 1/2 [[1,1],[1,-1]]", [inv(P), np.array([[-1, -1], [-1, 1]]) / -2], [np.array([[1, 1], [1, -1]]) / 2] * 2)
check("PD = [[3,1],[3,-1]]", P @ Dm, [[3, 1], [3, -1]])
check("PD P^-1 = 1/2 [[4,2],[2,4]] = A", [P @ Dm @ np.array([[1, 1], [1, -1]]), P @ Dm @ inv(P)], [[[4, 2], [2, 4]], A])
check("D^10 for D = diag(4,3)", np.linalg.matrix_power(np.diag([4, 3]), 10), np.diag([4 ** 10, 3 ** 10]))
for k in range(1, 11):
    check(f"A^{k} = P D^{k} P^-1 = 1/2[[3^k+1, 3^k-1],[3^k-1, 3^k+1]]", [np.linalg.matrix_power(A, k), P @ np.diag([3 ** k, 1]) @ inv(P)],
          [np.array([[3 ** k + 1, 3 ** k - 1], [3 ** k - 1, 3 ** k + 1]]) / 2] * 2)
# eigen visual presets
check("[[1,2],[2,4]]: eigenvalues 5, 0", sorted(np.linalg.eigvals(np.array([[1, 2], [2, 4]])), reverse=True), [5, 0])
ev = np.linalg.eigvals(Q)
check("Q has no real eigenvalues (|imag| = sin 45)", np.abs(ev.imag), [np.sqrt(0.5)] * 2)
check("[[4,7],[2,6]] eigenvalues 5 +- sqrt(15)", sorted(np.linalg.eigvals(np.array([[4, 7], [2, 6]])), reverse=True), [5 + np.sqrt(15), 5 - np.sqrt(15)])

# ---- 1.1.7 norms
x = np.array([3, 4])
check("||(3,4)||_1 = 7, ||(3,4)||_2 = 5", [np.abs(x).sum(), np.linalg.norm(x)], [7, 5])
x = np.array([3, -4])
check("x = [3,-4]: 7, 5, 4", [np.linalg.norm(x, 1), np.linalg.norm(x, 2), np.linalg.norm(x, np.inf)], [7, 5, 4])
check("||[[1,2],[3,4]]||_F = sqrt(30)", np.linalg.norm(np.array([[1, 2], [3, 4]]), "fro"), np.sqrt(30))
E = np.diag([1, 4])
check("x^T A x = 1 reaches x1 = 1 and x2 = 1/2", [np.array([1, 0]) @ E @ [1, 0], np.array([0, .5]) @ E @ [0, .5]], [1, 1])
for Lv, r in ((1, 1), (4, 2), (9, 3)):
    check(f"contour L={Lv} has radius {r}", r * r, Lv)
# l1-corner visual default: loss (w-c)^T H (w-c), H = R(20deg) diag(1,4) R^T, c = (2.2, 0.6), budget 1
th = np.radians(20)
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
Hm, c = R @ np.diag([1, 4]) @ R.T, np.array([2.2, 0.6])


def l1_solution(c, H, t):
    V = [np.array(p, float) * t for p in ((1, 0), (0, 1), (-1, 0), (0, -1))]
    best = None
    for k in range(4):
        p0, d = V[k], V[(k + 1) % 4] - V[k]
        s = np.clip(-(2 * d @ H @ (p0 - c)) / (2 * d @ H @ d), 0, 1)
        w = p0 + s * d
        f = (w - c) @ H @ (w - c)
        best = (f, w) if best is None or f < best[0] else best
    return best[1]


# brute force on a fine grid of the diamond boundary (independent of the edge formula)
ts = np.linspace(0, 4, 400001)
pts = np.array([np.interp(ts % 4, [0, 1, 2, 3, 4], [1, 0, -1, 0, 1]), np.interp(ts % 4, [0, 1, 2, 3, 4], [0, 1, 0, -1, 0])]).T
fv = np.einsum("ij,jk,ik->i", pts - c, Hm, pts - c)
check("l1-corner: L1 solution is the corner (1, 0)", l1_solution(c, Hm, 1), [1, 0])
check("l1-corner: brute force agrees", pts[fv.argmin()], [1, 0], tol=1e-4)
lo, hi = 0.0, 1e6
for _ in range(200):
    mu = (lo + hi) / 2
    w2 = np.linalg.solve(Hm + mu * np.eye(2), Hm @ c)
    lo, hi = (mu, hi) if np.linalg.norm(w2) > 1 else (lo, mu)
ang = np.linspace(0, 2 * np.pi, 2000001)
circ = np.c_[np.cos(ang), np.sin(ang)]
fc = np.einsum("ij,jk,ik->i", circ - c, Hm, circ - c)
check("l1-corner: L2 solution (0.979, 0.205)", w2, [0.9788, 0.2047], tol=5e-4)
check("l1-corner: L2 solution is on the circle", np.linalg.norm(w2), 1, tol=1e-9)
check("l1-corner: L2 brute force agrees", circ[fc.argmin()], w2, tol=1e-5)

# ---- 1.1.8 SVD of the 4x4 example
A4 = np.array([[4, 1, 2, 0], [1, 4, 0, 2], [2, 0, 4, 1], [0, 2, 1, 4]])
U, s, Vt = np.linalg.svd(A4)
check("singular values 7, 5, 3, 1", s, [7, 5, 3, 1])
v1 = np.full(4, 0.5)
check("A v1 = (3.5,3.5,3.5,3.5) = 7 * 0.5", A4 @ v1, [3.5] * 4)
check("every row of A sums to 7", A4.sum(1), [7] * 4)
vs = {5: np.array([1, -1, 1, -1]) / 2, 3: np.array([1, 1, -1, -1]) / 2, 1: np.array([1, -1, -1, 1]) / 2}
for sig, vv in vs.items():
    check(f"A v = {sig} v for the notes' v", A4 @ vv, sig * vv)
check("u1 v1^T = 0.25 everywhere", np.outer(v1, v1), np.full((4, 4), 0.25))
A1m = 7 * np.outer(v1, v1)
check("A1 = sigma1 u1 v1^T = 1.75 everywhere", A1m, np.full((4, 4), 1.75))
layers = [7 * np.outer(v1, v1)] + [sig * np.outer(vv, vv) for sig, vv in vs.items()]
check("A4 = A (all four layers)", sum(layers), A4)
for k in range(1, 5):
    Ak = sum(layers[:k])
    check(f"Eckart-Young k={k}: ||A - A_k||_F = sqrt(sum of discarded sigma^2)", np.linalg.norm(A4 - Ak, "fro"), np.sqrt(sum(x * x for x in [7, 5, 3, 1][k:])))
check("Eckart-Young values sqrt(35), sqrt(10), 1, 0", [np.sqrt(35), np.sqrt(10), 1, 0], [5.916079783, 3.16227766, 1, 0], tol=1e-8)
check("A^T A = V S^2 V^T", A4.T @ A4, Vt.T @ np.diag(s ** 2) @ Vt)
# svd2 visual presets (singular values shown)
for M, want in (([[2, 1], [1, 2]], [3, 1]), ([[1, 2], [3, 4]], None), ([[1, 2], [2, 4]], [5, 0]), (Q, [1, 1])):
    sv = np.linalg.svd(np.array(M, float), compute_uv=False)
    if want is not None:
        check(f"svd2 preset {np.round(M, 3).tolist()} singular values", sv, want)
    check(f"svd2 preset {np.round(M, 3).tolist()}: #nonzero sigma = rank", (sv > 1e-9).sum(), rank(np.array(M, float)))
check("[[1,2],[3,4]] singular values", np.linalg.svd(np.array([[1, 2], [3, 4]]), compute_uv=False), [5.4649857, 0.36596619], tol=1e-7)
done()
