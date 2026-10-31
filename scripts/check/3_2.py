"""Part 3.2 OLS, WLS & Ridge: the lecture's worked examples, the derivation identities and the visuals' defaults."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import check, done  # noqa: E402
from common_rng import rng, gauss  # noqa: E402

# quadratic-form expansion
A = np.array([[1.3, -0.4], [2.1, 0.7]]); b = np.array([0.6, -1.1])
a_, b_, c_, d_ = A.ravel()
check("beta^T A beta = a b1^2 + (b+c) b1 b2 + d b2^2", b @ A @ b, a_ * b[0] ** 2 + (b_ + c_) * b[0] * b[1] + d_ * b[1] ** 2)
check("gradient = [2a b1 + (b+c) b2, (b+c) b1 + 2d b2] = (A + A^T) beta", [2 * a_ * b[0] + (b_ + c_) * b[1], (b_ + c_) * b[0] + 2 * d_ * b[1]], (A + A.T) @ b)
# OLS example
X = np.array([[1, 1], [1, 2], [1, 3]]); y = np.array([1, 2, 2])
check("X^T X = [[3,6],[6,14]]", X.T @ X, [[3, 6], [6, 14]])
check("X^T y = [5,11]", X.T @ y, [5, 11])
check("(X^T X)^-1 = 1/6 [[14,-6],[-6,3]]", np.linalg.inv(X.T @ X), np.array([[14, -6], [-6, 3]]) / 6)
bh = np.linalg.solve(X.T @ X, X.T @ y)
check("beta-hat ~ (0.6667, 0.5)", np.round(bh, 4), [0.6667, 0.5])
e = y - X @ bh
check("residual is perpendicular to 1 and x", X.T @ e, [0, 0])
P = X @ np.linalg.inv(X.T @ X) @ X.T
check("P_X y = y-hat", P @ y, X @ bh)
check("P_X^2 = P_X", P @ P, P)
# WLS: W = I gives OLS; rebuild the visual's heteroscedastic data and compare OLS / WLS
check("W = I: WLS = OLS", np.linalg.solve(X.T @ np.eye(3) @ X, X.T @ np.eye(3) @ y), bh)
r = rng(61)
H = []
for _ in range(30):
    x = r() * 10; s = 0.3 + 0.35 * x
    H.append((x, 1 + 0.8 * x + s * gauss(r), s))
H = np.array(H); Xh = np.c_[np.ones(30), H[:, 0]]; W = np.diag(1 / H[:, 2] ** 2)
bo = np.linalg.lstsq(Xh, H[:, 1], rcond=None)[0]
bw = np.linalg.solve(Xh.T @ W @ Xh, Xh.T @ W @ H[:, 1])
print(f"   (wls visual: OLS {bo.round(3).tolist()}, WLS {bw.round(3).tolist()})")
e3, w3 = np.array([0.5, -1.0, 2.0]), np.array([2.0, 1.0, 0.5])
check("e^T W e = sum w_i e_i^2", e3 @ np.diag(w3) @ e3, (w3 * e3 ** 2).sum())
# collinearity intuition and 2x2 lift
check("(1,0) and (100,-99) give the same y-hat when x1 = x2", [1 + 0, 100 - 99], [1, 1])
M = np.array([[4, 3.99], [3.99, 4]])
check("eigenvalues of X^T X + I are lambda_i + 1", np.linalg.eigvalsh(M + np.eye(2)), np.linalg.eigvalsh(M) + 1)
# ridge example
RX = np.array([[1, 1, 1], [1, 1, .9], [1, .9, 1], [1, .9, .9]]); RY = np.array([2.8, 2.5, 2.6, 2.4])
G = RX.T @ RX
check("X^T X as printed", G, [[4, 3.8, 3.8], [3.8, 3.62, 3.61], [3.8, 3.61, 3.62]])
check("X^T y as printed", RX.T @ RY, [10.3, 9.8, 9.81])
check("det(X^T X) = 0.0004", np.linalg.det(G), 0.0004, tol=1e-12)
check("OLS ~ (-1.225, 1.500, 2.500)", np.linalg.solve(G, RX.T @ RY), [-1.225, 1.5, 2.5])
br = np.linalg.solve(G + np.eye(3), RX.T @ RY)
check("ridge lambda = 1 ~ (0.835, 0.801, 0.811)", np.round(br, 3), [0.835, 0.801, 0.811])
check("lambda -> 0: ridge -> OLS; lambda -> inf: ridge -> 0", [np.linalg.solve(G + 1e-9 * np.eye(3), RX.T @ RY), np.linalg.solve(G + 1e9 * np.eye(3), RX.T @ RY)], [[-1.225, 1.5, 2.5], [0, 0, 0]], tol=1e-5)
done()
