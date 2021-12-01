import sympy as sp

from utils.expressions import grad, laplace

x, y, pi, nu = sp.var("x y pi nu")

def get_u():
    return sp.sin(pi * x) * sp.sin(pi * y)


def get_f(u):
    return - nu * laplace(u)


u = get_u()
u_x, u_y = grad(u)

print("u =", u)
print("u_x=", u_x)
print("u_y=", u_y)

print("f =", get_f(u))
