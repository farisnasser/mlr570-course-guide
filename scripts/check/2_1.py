"""Part 2.1 Data Preprocessing: the notes' numbers and the visuals' defaults (datasets rebuilt with the same seeded RNG)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import check, done  # noqa: E402
from common_rng import rng, gauss  # noqa: E402

check("sqrt(1000^2 + 5^2) ~ 1000.01", round(math.hypot(1000, 5), 2), 1000.01)
S = np.array([2, 3, 3, 4, 100])
check("{2,3,3,4,100}: mean 112/5 = 22.4, median 3", [S.sum(), S.mean(), np.median(S)], [112, 22.4, 3])
ages = np.array([20, 30, 40, 60])
check("min-max: 30 -> 0.25", (30 - 20) / (60 - 20), 0.25)
check("with 100: 30 -> 10/80 = 0.125", (30 - 20) / (100 - 20), 0.125)
z = (ages - ages.mean()) / ages.std()
check("z-scores have mean 0, std 1 on the fitting data", [z.mean(), z.std()], [0, 1])
check("one-hot: e_i^T e_j = 0 and ||e_i - e_j|| = sqrt 2", [np.eye(3)[0] @ np.eye(3)[1], np.linalg.norm(np.eye(3)[0] - np.eye(3)[2])], [0, math.sqrt(2)])
check("label encoding |0-2| = 2, |0-1| = 1", [abs(0 - 2), abs(0 - 1)], [2, 1])
check("binary columns for k = 1000: ceil(log2 1000) = 10, 2^10 = 1024", [math.ceil(math.log2(1000)), 2 ** 10], [10, 1024])
# rolling windows (right-aligned, first w-1 undefined)
x = np.arange(10.0)
w = 3
roll = [None if t < w - 1 else x[t - w + 1:t + 1].mean() for t in range(10)]
check("rolling mean of 0..9, w=3: undefined twice, then t-1", [roll[2], roll[9], sum(r is None for r in roll)], [1, 8, 2])
# stratified split of 400 with 20 positives into 80: exactly 4 positives = 5%
check("stratified 20% split of 5% positives keeps 5%", round(20 * 80 / 400) / 80, 0.05)

# binning on the visual's 14 values
X = np.array([1, 1.5, 2, 2.5, 3, 3.3, 3.8, 4.6, 5.5, 7.5, 9.5, 12, 16, 25])
e = np.linspace(X.min(), X.max(), 5)
cnt = [((X >= e[0]) & (X <= e[1])).sum()] + [((X > e[i]) & (X <= e[i + 1])).sum() for i in range(1, 4)]
check("equal-width counts 9, 3, 1, 1", cnt, [9, 3, 1, 1])
check("equal-frequency counts 4, 4, 3, 3", [len(a) for a in np.array_split(np.sort(X), 4)], [4, 4, 3, 3])
check("width = (25 - 1)/4 = 6", (X.max() - X.min()) / 4, 6)

# PCA cloud: rebuild PCA_DATA, check Sigma, eigen, and that the eigenvector maximizes projected variance
r = rng(17)
P = []
for _ in range(40):
    a, b = gauss(r) * 1.6, gauss(r) * 0.55
    c, s = math.cos(0.6), math.sin(0.6)
    P.append([0.4 + c * a - s * b, 0.3 + s * a + c * b])
P = np.array(P)
Xc = P - P.mean(0)
Sig = Xc.T @ Xc / (len(P) - 1)
check("Sigma = cov(X) with n-1", Sig, np.cov(P.T))
lam, V = np.linalg.eigh(Sig)
ang = np.linspace(0, np.pi, 200001)
pv = np.cos(ang) ** 2 * Sig[0, 0] + 2 * np.cos(ang) * np.sin(ang) * Sig[0, 1] + np.sin(ang) ** 2 * Sig[1, 1]
check("max projected variance = lambda_1", pv.max(), lam[-1], tol=1e-9)
check("min projected variance = lambda_2", pv.min(), lam[0], tol=1e-9)
U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
check("PCA via SVD: lambda_i = sigma_i^2/(n-1)", sorted(sv ** 2 / (len(P) - 1)), sorted(lam))
check("W^T Sigma W is diagonal", (V.T @ Sig @ V)[0, 1], 0)
print(f"   (PCA visual: lambda = {lam[::-1].round(4).tolist()}, EVR1 = {100 * lam[-1] / lam.sum():.1f}%)")
check("EVR example 70, 20, 10: PC1+PC2 = 90%", (70 + 20) / 100, 0.9)
# scale visual: rebuild SCALE_DATA; nearest neighbour of the default query changes after scaling
r = rng(11)
D = np.array([[5000 + r() * 10000, 1 + r() * 18] for _ in range(16)])
q = np.array([6000, 2])
nn = lambda T: int(np.argmin(np.linalg.norm(T(D) - T(q[None]), axis=1)))
raw = nn(lambda A: A)
mm = nn(lambda A: (A - D.min(0)) / (D.max(0) - D.min(0)))
zz = nn(lambda A: (A - D.mean(0)) / D.std(0))
print(f"   (scale visual nearest: raw {raw}, min-max {mm}, z {zz})")
check("scaling changes the nearest neighbour of the default query", raw != mm or raw != zz, 1)
done()
