"""Part 3.1 Hard-Margin SVM: the notes' geometry examples, the toy dual, and the worked example solved as a real QP."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import check, done  # noqa: E402

check("(3,1) - (1,3) = (2,-2); (1,1).(2,-2) = 0", [np.subtract((3, 1), (1, 3)), [np.dot((1, 1), (2, -2)), 0]], [[2, -2], [0, 0]])
check("x+y-4: (3,3) -> 2, (0,0) -> -4", [3 + 3 - 4, -4], [2, -4])
check("distance (0,0) to x+y-4=0 = 4/sqrt2 = 2 sqrt2 ~ 2.83", [4 / math.sqrt(2), round(2 * math.sqrt(2), 2)], [2 * math.sqrt(2), 2.83])
w, b, x0 = np.array([1.0, 1]), -1.0, np.array([0.3, 2.0])
check("scaling (w, b) by 10 leaves the distance unchanged", abs(w @ x0 + b) / np.linalg.norm(w), abs(10 * w @ x0 + 10 * b) / np.linalg.norm(10 * w))
z = x0 - (w @ x0 + b) / (w @ w) * w
check("foot z lies on the line and x0 - z is parallel to w", [w @ z + b, (x0 - z)[0] * w[1] - (x0 - z)[1] * w[0]], [0, 0])
# toy dual
g = lambda a: a - a * a / 4
check("g(0), g(1), g(2), g(3) = 0, 0.75, 1, 0.75", [g(0), g(1), g(2), g(3)], [0, 0.75, 1, 0.75])
aa = np.linspace(0, 6, 600001)
check("max g = 1 = p* at alpha = 2", [g(aa).max(), aa[np.argmax(g(aa))]], [1, 2], tol=1e-5)
for a in (0.5, 1.7, 3.2):
    xs = np.linspace(-5, 5, 200001)
    check(f"g({a}) = min_x x^2 - a(x-1)", (xs ** 2 - a * (xs - 1)).min(), g(a), tol=1e-8)
# worked example: solve the hard-margin primal QP with a generic solver (SLSQP) and compare with the lecture solution
D = np.array([[2, 2, 1], [3, 1.6, 1], [2.6, 2.6, 1], [3, 2.6, 1], [2, 3, 1], [-1, -1, -1], [0, -2, -1], [-2, 0, -1], [-1.4, -1.2, -1], [0.6, -1.6, -1]])
X, y = D[:, :2], D[:, 2]
try:
    from scipy.optimize import minimize
    res = minimize(lambda v: 0.5 * (v[0] ** 2 + v[1] ** 2), x0=[1, 1, 0], constraints=[{"type": "ineq", "fun": lambda v: y * (X @ v[:2] + v[2]) - 1}], method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    check("SLSQP primal = (8/29, 13/29, -13/29)", res.x, [8 / 29, 13 / 29, -13 / 29], tol=1e-6)
except ImportError:
    print("   (scipy not available: QP cross-check skipped; KKT check below still proves optimality)")
ws, bs = np.array([8, 13]) / 29, -13 / 29
marg = y * (X @ ws + bs)
check("w* ~ (0.276, 0.448), b* ~ -0.448", np.round([*ws, bs], 3), [0.276, 0.448, -0.448])
check("||w*|| = sqrt(233)/29 ~ 0.526; 2/||w*|| ~ 3.80", [round(np.linalg.norm(ws), 3), round(2 / np.linalg.norm(ws), 2), math.sqrt(233) / 29], [0.526, 3.80, np.linalg.norm(ws)])
check("support vectors (2,2), (-2,0), (0.6,-1.6) give exactly 1", marg[[0, 7, 9]], [1, 1, 1])
check("table: 31.8/29 ~ 1.10, 34/29 ~ 1.17", [round(31.8 / 29, 2), round(34 / 29, 2), round(marg[1], 2), round(marg[5], 2)], [1.10, 1.17, 1.10, 1.17])
check("every other point > 1", np.all(np.delete(marg, [0, 7, 9]) > 1), 1)
# KKT: alpha >= 0 on the SVs with w = sum alpha y x and sum alpha y = 0 -> the lecture solution is optimal
S = [0, 7, 9]
A = np.vstack([(y[S] * X[S].T), y[S]])
alpha = np.linalg.solve(A, [*ws, 0])
print("   (alphas on the support vectors:", np.round(alpha, 4).tolist(), ")")
check("alphas on the support vectors are positive (KKT)", np.all(alpha > 0), 1)
done()
