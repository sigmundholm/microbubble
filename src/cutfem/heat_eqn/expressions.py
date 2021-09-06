import sympy as sp
from utils.expressions import *

t, x, y, pi = sp.var("t x y pi")
nu, tau = sp.var("nu tau")


def get_u():
    return sp.sin(pi * x) * sp.sin(pi * y) * (2 + sp.sin(pi * t))


def get_f(u):
    return sp.diff(u, t) - nu * laplace(u)


def get_f_stationary(u):
    u0 = u.subs(t, 0)
    u1 = u.subs(t, 1)
    return (u1 - u0)/tau - 0.5 * nu * (laplace(u1) + laplace(u0))


u = get_u()
f = get_f(u)

print("u =", u)
print(" f =", f)
print("stat f =", get_f_stationary(u))
