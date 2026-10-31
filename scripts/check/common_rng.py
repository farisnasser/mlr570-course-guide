"""Python port of src/viz/stats.js rng() (mulberry32) and gauss(), so check scripts can rebuild the visuals' datasets exactly."""
import math


def rng(seed=1):
    a = seed & 0xFFFFFFFF

    def imul(x, y):
        return ((x & 0xFFFFFFFF) * (y & 0xFFFFFFFF)) & 0xFFFFFFFF

    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = imul(t ^ (t >> 15), t | 1)
        t ^= (t + imul(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return nxt


def gauss(r):
    u = 0.0
    while u == 0:
        u = r()
    return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * r())
