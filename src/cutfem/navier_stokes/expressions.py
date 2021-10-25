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


u = get_u(t)
p = get_p(t)
f1, f2 = get_f(u, p)
print("f_1 =", f1)
print("f_2 =", f2)

print("Simplified")
print("f_1 =", sp.simplify(f1))
print("f_2 =", sp.simplify(f2))
