import sympy as sp

from utils.expressions import laplace, grad, get_u, get_p

x, y, t, nu, pi = sp.var("x y t nu, pi")


def get_f(u, p):
    u1, u2 = u
    p1, p2 = grad(p)
    u1_x, u1_y = grad(u1)
    u2_x, u2_y = grad(u2)
    f1 = u1_x * u1 + u1_y * u2 - nu * laplace(u1) + p1
    f2 = u2_x * u1 + u2_y * u2 - nu * laplace(u2) + p2
    return f1, f2


u = get_u(0)
u1, u2 = u
p = get_p(0)

f1, f2 = get_f(u, p)

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
