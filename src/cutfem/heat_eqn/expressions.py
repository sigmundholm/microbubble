import sympy as sp
from utils.expressions import *

t, x, y, pi = sp.var("t x y pi")
nu, tau = sp.var("nu tau")


def get_u():
    return sp.sin(pi * x) * sp.sin(pi * y) * sp.exp(-t)


def get_f(u):
    return sp.diff(u, t) - nu * laplace(u)


u = get_u()
f = get_f(u)

print("u =", u)
print(" f =", f)
