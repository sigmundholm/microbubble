import sympy as sp

from utils.expressions import laplace, grad, get_u, get_p

x, y, t, nu, pi = sp.var("x y t nu, pi")


def get_f(u, p):
    u1, u2 = u
    p1, p2 = grad(p)
    u1_x, u1_y = grad(u1)
    u2_x, u2_y = grad(u2)
    f1 = sp.diff(u1, t) + u1_x * u1 + u1_y * u2 - nu * laplace(u1) + p1
    f2 = sp.diff(u2, t) + u2_x * u1 + u2_y * u2 - nu * laplace(u2) + p2
    return f1, f2

def get_conv_f(u, p, conv_field):
    u1, u2 = u
    b1, b2 = conv_field
    p1, p2 = grad(p)
    u1_x, u1_y = grad(u1)
    u2_x, u2_y = grad(u2)
    f1 = sp.diff(u1, t) + u1_x * b1 + u1_y * b2 - nu * laplace(u1) + p1
    f2 = sp.diff(u2, t) + u2_x * b1 + u2_y * b2 - nu * laplace(u2) + p2
    return f1, f2

def get_conv_field():
    return sp.cos(pi * x), sp.sin(pi * y)

def get_u2(t):
    """Ethier-Steinman (1994): modified time dependency."""
    print("(compute u)")
    time = sp.cos(t)
    u1 = -sp.cos(pi * x) * sp.sin(pi * y) * time
    u2 = sp.sin(pi * x) * sp.cos(pi * y) * time
    return u1, u2


def get_p2(t=0):
    """Ethier-Steinman (1994): modified time dependency."""
    print("(compute p)")
    time = sp.cos(t)
    return -(sp.cos(2 * sp.pi * x) + sp.cos(2 * sp.pi * y)) / 4 * time


u = get_u2(t)
u1, u2 = u
p = get_p2(t)
conv_field = get_conv_field()

f1, f2 = get_conv_f(u, p, u)

print("\nf =")
print("val[0] =", sp.simplify(f1), ";")
print("val[1] =", sp.simplify(f2), ";")

print("\nu =")
print("div u =", sp.simplify(sp.diff(u1, x) + sp.diff(u2, y)))
print("val[0] =", u[0], ";")
print("val[1] =", u[1], ";")

print("\ngrad u")
u1_x, u1_y = grad(u1)
u2_x, u2_y = grad(u2)
print("value[0][0] =", u1_x, ";")
print("value[0][1] =", u1_y, ";")
print("value[1][0] =", u2_x, ";")
print("value[1][1] =", u2_y, ";")

print("\nPressure")
p_x, p_y = grad(p)
print("p =", p, ";")
print("\nvalue[0] =", p_x, ";")
print("value[1] =", p_y, ";")


