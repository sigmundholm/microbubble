import sympy as sp
from utils.expressions import *

x, y, pi = sp.var("x y pi")
nu, tau = sp.var("nu tau")


def get_u():
    return sp.sin(pi * x) * sp.sin(pi * y)


def get_f(u):
    return u / tau - nu * laplace(u)


u = get_u()
f = get_f(u)

print("u =", u)
print(" f =", f)
