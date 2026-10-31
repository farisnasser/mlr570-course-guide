"""Part 1.2 Probability & Statistics: every number in the notes' worked examples and the visuals' defaults."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import check, done  # noqa: E402

# 1.2.1 PMF / CDF example
p = np.array([0.25, 0.5, 0.25])
check("PMF sums to 1", p.sum(), 1)
check("F(1) = 1/4 + 1/2 = 3/4", p[:2].sum(), 0.75)
check("CDF steps 1/4, 3/4, 1", np.cumsum(p), [0.25, 0.75, 1])
# pdf-area visual: standard normal mu +- k sigma
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
check("Normal: within 1 sigma = 0.6827 (about 68%)", Phi(1) - Phi(-1), 0.682689492, tol=1e-9)
check("Normal: within 2 sigma = 0.9545 (about 95%)", Phi(2) - Phi(-2), 0.954499736, tol=1e-9)
check("Normal: within 3 sigma = 0.9973 (about 99.7%)", Phi(3) - Phi(-3), 0.997300204, tol=1e-9)
xs = np.linspace(-1, 1, 200001)
check("area under the pdf on [-1,1] equals F(1) - F(-1)", np.trapezoid(np.exp(-xs ** 2 / 2) / math.sqrt(2 * math.pi), xs), Phi(1) - Phi(-1), tol=1e-9)
check("Exponential(1): P(0.5 <= X <= 2) = e^-0.5 - e^-2", (1 - math.exp(-2)) - (1 - math.exp(-0.5)), math.exp(-0.5) - math.exp(-2))

# 1.2.2 moments
A, B = np.array([9, 10, 11]), np.array([0, 10, 20])
check("both means 10", [A.mean(), B.mean()], [10, 10])
check("deviations of A cancel", (A - 10).mean(), 0)
check("Var(A) = 2/3, Var(B) = 200/3", [A.var(), B.var()], [2 / 3, 200 / 3])
check("sigma_A ~ 0.82, sigma_B ~ 8.16", [round(A.std(), 2), round(B.std(), 2)], [0.82, 8.16])
X = np.array([0, 1, 2])
check("Var = E[X^2] - E[X]^2 on the PMF example (mu 1, var 0.5)", [(X * p).sum(), (X ** 2 * p).sum() - (X * p).sum() ** 2], [1, 0.5])
check("Bernoulli(0.3): E = 0.3, Var = 0.3 - 0.09 = 0.21", [0.3, 0.3 - 0.09], [0.3, 0.21])
s = np.array([3, 5, 2, 4])
check("sample (3,5,2,4): mean 3.5, S^2 = 5/3", [s.mean(), s.var(ddof=1)], [3.5, 5 / 3])
check("S^2 ~ 1.67", round(s.var(ddof=1), 2), 1.67)
# sampling visual: E[S^2 with n-1] = sigma^2 and E[S^2 with n] = (n-1)/n sigma^2, Var(Xbar) = sigma^2/n  (exact, by enumeration for n = 4)
import itertools
n = 4
probs, s2a, s2b, xb = [], [], [], []
for t in itertools.product([0, 1, 2], repeat=n):
    w = np.prod(p[list(t)]); a = np.array(t)
    probs.append(w); s2a.append(a.var(ddof=1)); s2b.append(a.var()); xb.append(a.mean())
probs = np.array(probs)
check("E[S_n^2] = sigma^2 exactly (n = 4)", probs @ np.array(s2a), 0.5)
check("dividing by n gives (n-1)/n sigma^2", probs @ np.array(s2b), 0.375)
check("Var(Xbar_n) = sigma^2/n", probs @ (np.array(xb) - 1) ** 2, 0.5 / 4)

# 1.2.3 80 students
N = np.array([[36, 14], [9, 21]])
check("totals 50, 30 / 45, 35 / 80", [N.sum(1), N.sum(0), [N.sum(), 0]], [[50, 30], [45, 35], [80, 0]])
check("P(passed & studied) = 36/80, P(passed) = 50/80, P(failed|studied) = 9/45", [36 / 80, 50 / 80, 9 / 45], [N[0, 0] / 80, N[0].sum() / 80, N[1, 0] / N[:, 0].sum()])
check("P(X=1) = 36/80 + 9/80 = 45/80", (36 + 9) / 80, 45 / 80)
check("conditional on studied sums to 1", N[:, 0].sum() / 45, 1)

# 1.2.4 conditional example and Bayes
check("0.4/0.6 = 2/3 and 1/3, sum 1", [0.4 / 0.6, 1 - 0.4 / 0.6, 2 / 3 + 1 / 3], [2 / 3, 1 / 3, 1])
pp = 0.95 * 0.01 + 0.05 * 0.99
check("P(+) = 0.059", pp, 0.059)
check("P(D|+) ~ 0.161", round(0.95 * 0.01 / pp, 3), 0.161)
check("Bayes counts of 10,000: 95 sick positives, 495 healthy positives", [round(0.95 * 100), round(0.05 * 9900)], [95, 495])

# joint-density visual: bivariate normal (standard marginals, rho); marginal and conditional integrate correctly
def fxy(x, y, r):
    return np.exp(-(x * x - 2 * r * x * y + y * y) / (2 * (1 - r * r))) / (2 * np.pi * np.sqrt(1 - r * r))
g = np.linspace(-9, 9, 6001)
for r in (0.6, 0.0, -0.8):
    check(f"rho={r}: integral over x of f(x, 0.8) = f_Y(0.8)", np.trapezoid(fxy(g, 0.8, r), g), math.exp(-0.32) / math.sqrt(2 * math.pi), tol=1e-9)
check("rho=0: f(x,y) = f_X(x) f_Y(y)", fxy(0.3, -1.1, 0.0), math.exp(-0.045) / math.sqrt(2 * math.pi) * math.exp(-0.605) / math.sqrt(2 * math.pi))

# 1.2.5 independence table
P = np.array([[0.15, 0.15], [0.35, 0.35]])
check("marginals 0.3, 0.7 and 0.5, 0.5", [P.sum(1), P.sum(0)], [[0.3, 0.7], [0.5, 0.5]])
check("every cell = product of marginals", P, np.outer(P.sum(1), P.sum(0)))
check("two fair coins: 1/4 = 1/2 * 1/2", 0.25, 0.5 * 0.5)

# 1.2.6 / 1.2.7 E[XY] example
P = np.array([[0.2, 0.3], [0.1, 0.4]]); xv = yv = np.array([1, 2])
EXY = (np.outer(xv, yv) * P).sum()
check("E[XY] = 0.2 + 0.6 + 0.2 + 1.6 = 2.6", [EXY, 0.2 + 0.6 + 0.2 + 1.6], [2.6, 2.6])
px, py = P.sum(1), P.sum(0)
check("P(X) = 0.5, 0.5; P(Y) = 0.3, 0.7", [px, py], [[0.5, 0.5], [0.3, 0.7]])
EX, EY = xv @ px, yv @ py
check("E[X] = 1.5, E[Y] = 1.7", [EX, EY], [1.5, 1.7])
check("Cov = 2.6 - 2.55 = 0.05", EXY - EX * EY, 0.05)
VX, VY = xv ** 2 @ px - EX ** 2, yv ** 2 @ py - EY ** 2
check("Var(X) = 2.5 - 2.25 = 0.25, Var(Y) = 3.1 - 2.89 = 0.21", [VX, VY, xv ** 2 @ px, yv ** 2 @ py], [0.25, 0.21, 2.5, 3.1])
check("sigma_Y ~ 0.458", round(math.sqrt(VY), 3), 0.458)
rho = (EXY - EX * EY) / math.sqrt(VX * VY)
check("rho ~ 0.22", round(rho, 2), 0.22)
check("E[X+Y] = E[X] + E[Y]", (np.add.outer(xv, yv) * P).sum(), EX + EY)
# cov-scatter preset B = A^2 on A = -2..2 has Cov 0 (and the line B = 2A has rho = 1)
a = np.arange(-2, 3)
check("B = A^2: Cov = (-8-1+0+1+8)/5 - 0 = 0", np.mean(a * a ** 2) - a.mean() * np.mean(a ** 2), 0)
check("B = 2A: rho = 1", np.corrcoef(a, 2 * a)[0, 1], 1)

# 1.2.8 total expectation
check("E[X] = 0.75(80) + 0.25(60) = 75", 0.75 * 80 + 0.25 * 60, 75)
done()
