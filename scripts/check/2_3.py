"""Part 2.3 Model Evaluation: the lecture numbers, plus the visuals' fits and counts rebuilt from the same seeds."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import check, done  # noqa: E402
from common_rng import rng, gauss  # noqa: E402

check("split 70/15/15 of 1000 = 700/150/150", [0.7 * 1000, 0.15 * 1000, 0.15 * 1000], [700, 150, 150])
tr = [12, 17, 6, 23, 11]; va = [11, 18, 19, 23.5, 17]
check("model 1 has the lowest validation error, model 3 the lowest training error", [int(np.argmin(va)) + 1, int(np.argmin(tr)) + 1], [1, 3])
check("accuracy warning: 95/100 = 95%, recall 0/5 = 0", [95 / 100, 0 / 5], [0.95, 0])


def prf(TP, FN, FP, TN, beta=1):
    P, R = TP / (TP + FP), TP / (TP + FN)
    return P, R, 2 * P * R / (P + R), (1 + beta ** 2) * P * R / (beta ** 2 * P + R)


check("F1(0.1, 0.9) = 0.18; GM 0.3; AM 0.5", [2 * .1 * .9 / 1.0, math.sqrt(0.09), 0.5], [0.18, 0.3, 0.5])
for row, want in (((50, 10, 10, 50), (0.833, 0.833, 0.833)), ((10, 40, 2, 48), (0.833, 0.200, 0.323)), ((45, 5, 40, 10), (0.529, 0.900, 0.667))):
    P, R, F1, _ = prf(*row)
    check(f"lecture row {row}: P, R, F1 (3 dp)", [round(P, 3), round(R, 3), round(F1, 3)], want)
P, R, F1, _ = prf(10, 40, 2, 48)
print(f"   NOTES DISCREPANCY (reported, not changed): 'High P, low R' F1 = {F1:.6f}; the lecture table prints 0.320")
P, R, F1, _ = prf(45, 5, 40, 10)
print(f"   note: 'High R, low P' F1 = {F1:.6f}; the lecture table prints 0.666 (truncated, rounds to 0.667)")
P, R, F1, F2 = prf(45, 40, 5, 10, beta=2)
check("F-beta example: P = 0.9, R = 45/85 = 0.529, F1 ~ 0.666", [P, round(R, 3), math.floor(F1 * 1000) / 1000], [0.9, 0.529, 0.666])
check("F2 = 5(0.9)(0.529)/(4(0.9)+0.529) = 2.38/4.13 = 0.577", [round(5 * 0.9 * 0.529, 2), round(4 * 0.9 + 0.529, 2), round(F2, 3)], [2.38, 4.13, 0.577])
y, yh = np.array([3, 0.5, 2, 7]), np.array([2.5, 0, 2, 8])
check("example 1: MSE 0.375, MAE 0.5, RMSE ~ 0.61", [np.mean((y - yh) ** 2), np.mean(np.abs(y - yh)), round(math.sqrt(0.375), 2)], [0.375, 0.5, 0.61])
y, yh = np.append(y, 100), np.append(yh, 8)
check("example 2: 92^2 = 8464, MSE 1693.1, MAE 18.8", [92 ** 2, np.mean((y - yh) ** 2), np.mean(np.abs(y - yh))], [8464, 1693.1, 18.8])
c = np.linspace(-5, 110, 115001)
check("best constant for MSE = mean, for MAE = median", [c[np.argmin([np.mean((y - k) ** 2) for k in c[::50]]) * 50], c[np.argmin([np.mean(np.abs(y - k)) for k in c[::50]]) * 50]], [y.mean(), np.median(y)], tol=0.03)

# complexity visual: rebuild DATA and the error curves; validation error has an interior minimum and training error falls
truth = lambda x: math.sin(2.2 * x) + 0.3 * x
r = rng(31)
def draw(n):
    out = []
    for _ in range(n):
        x = r() * 2 - 1
        out.append((x, truth(x) + 0.25 * gauss(r)))
    return out
train, val = draw(14), draw(40)
tx, ty = np.array(train).T
vx, vy = np.array(val).T
curves = []
for d in range(13):
    V = np.vander(tx, d + 1, increasing=True)
    b = np.linalg.lstsq(V, ty, rcond=None)[0]
    curves.append((np.mean((V @ b - ty) ** 2), np.mean((np.vander(vx, d + 1, increasing=True) @ b - vy) ** 2)))
curves = np.array(curves)
best = int(np.argmin(curves[:, 1]))
print(f"   (complexity visual: best degree {best}; train/val at 1: {curves[1].round(4).tolist()}, at 12: {curves[12].round(4).tolist()})")
check("training error never rises with degree", np.all(np.diff(curves[:, 0]) <= 1e-9), 1)
check("validation minimum is interior (under- and overfitting both visible)", 0 < best < 12 and curves[12, 1] > 2 * curves[best, 1], 1)

# threshold visual: rebuild SCORES; counts at t = 0.5 add up and the PR trade-off moves the right way
r = rng(51)
cl = lambda v: max(0.005, min(0.995, v))
pos = [cl(0.64 + 0.13 * gauss(r)) for _ in range(100)]
neg = [cl(0.36 + 0.13 * gauss(r)) for _ in range(100)]
cnt = lambda t: (sum(s >= t for s in pos), sum(s < t for s in pos), sum(s >= t for s in neg), sum(s < t for s in neg))
TP, FN, FP, TN = cnt(0.5)
print(f"   (threshold visual at t=0.5: TP {TP}, FN {FN}, FP {FP}, TN {TN})")
check("counts add to 100 + 100", [TP + FN, FP + TN], [100, 100])
lo, hi = cnt(0.3), cnt(0.7)
check("lower t: recall up, precision down; higher t: the reverse", [lo[0] / 100 >= TP / 100 >= hi[0] / 100, lo[0] / (lo[0] + lo[2]) <= TP / (TP + FP) <= hi[0] / (hi[0] + hi[2])], [1, 1])
done()
