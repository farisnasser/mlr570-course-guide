"""Part 2.2 Similarity Measures: the notes' numbers and the curse-of-dimensionality simulation trend."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import check, done  # noqa: E402
from common_rng import rng  # noqa: E402

A, B = np.array([1, 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10])
check("B = 2A: rho = 1", np.corrcoef(A, B)[0, 1], 1)
A = np.array([-2, -1, 0, 1, 2]); B = A ** 2
check("B = A^2: E[AB] = (-8-1+0+1+8)/5 = 0, Cov = 0, rho = 0", [np.mean(A * B), np.mean(A * B) - A.mean() * B.mean(), np.corrcoef(A, B)[0, 1]], [0, 0, 0])
a, b = np.array([1, 2, 3]), np.array([4, 5, 6])
check("Euclidean sqrt(27) ~ 5.196", [math.sqrt(27), round(np.linalg.norm(a - b), 3)], [np.linalg.norm(a - b), 5.196])
check("Manhattan 9", np.abs(a - b).sum(), 9)
check("[1,1,10]: Euclidean ~ 10.1, Manhattan 12", [round(math.sqrt(102), 1), 12], [10.1, 1 + 1 + 10])
C = 2 * a
check("cos(A, 2A) = 28/(sqrt14 sqrt56) = 1", [a @ C, math.sqrt(14) * math.sqrt(56), a @ C / (np.linalg.norm(a) * np.linalg.norm(C))], [28, 28, 1])
check("||A - 2A|| = sqrt(14) ~ 3.74", round(np.linalg.norm(a - C), 2), 3.74)
x, y = np.array([3.0, 1, 4, 1, 5]), np.array([2.0, 7, 1, 8, 2])
xc, yc = x - x.mean(), y - y.mean()
check("Pearson = cosine of mean-centred vectors", xc @ yc / (np.linalg.norm(xc) * np.linalg.norm(yc)), np.corrcoef(x, y)[0, 1])
SA, SB = {1, 2, 3, 4, 5}, {4, 5, 6, 7}
check("J = 2/7 ~ 0.286", [len(SA & SB) / len(SA | SB), round(2 / 7, 3)], [2 / 7, 0.286])
va, vb = np.array([1, 0, 1, 0, 1]), np.array([1, 1, 0, 0, 1])
check("binary J = 2/4 = 0.5", ((va & vb).sum() / (va | vb).sum()), 0.5)


# curse visual: same simulation as parts/2.2/viz.js distStats (seed + d*7919, query first, then 1000 points)
def dist_stats(d, seed=1, n=1000, metric="l2"):
    r = rng(seed + d * 7919)
    q = [r() for _ in range(d)]
    ds = []
    for _ in range(n):
        z = [r() - qj for qj in q]
        ds.append(math.sqrt(sum(v * v for v in z)) if metric == "l2" else sum(abs(v) for v in z))
    ds = np.array(ds)
    return ds.min(), ds.max(), (ds.max() - ds.min()) / ds.min()


con = {d: dist_stats(d)[2] for d in (1, 10, 100, 1000)}
print("   (curse visual contrast, Euclidean):", {d: round(c, 3) for d, c in con.items()})
check("contrast falls as d grows (10 -> 100 -> 1000)", [con[10] > con[100] > con[1000]], [1])
check("at d = 1000 contrast < 0.3", con[1000] < 0.3, 1)
done()
