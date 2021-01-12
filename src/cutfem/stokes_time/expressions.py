import sympy as sp
from sympy import lambdify, simplify
from utils.expressions import *


def get_f(u, p):
    t, nu = sp.var("t nu")
    u1, u2 = u
    p_x, p_y = grad(p)
    f1 = sp.diff(u1, t) - nu * laplace(u1) + p_x
    f2 = sp.diff(u2, t) - nu * laplace(u2) + p_y
    return f1, f2


def get_f2(u):
    """Calculate the rhs as the convection term (u solves the homogeneous NS)."""
    c1, c2 = convection(u)
    return [-c1, -c2]


def get_f_stokes_gen(u, p):
    u1, u2 = u
    p_x, p_y = grad(p)
    f1 = (delta * u1 - tau * nu * laplace(u1) + tau * p_x - u1) / tau
    f2 = (delta * u2 - tau * nu * laplace(u2) + tau * p_y - u2) / tau
    return f1, f2


if __name__ == '__main__':
    """
    Here we want to solve the equations
        
        ∂_t u - νΔu + ∇p = f
    """
    t, nu, tau = sp.var("t nu tau")
    delta = sp.var("delta")

    u = get_u(t)
    u1, u2 = u
    p = get_p(t)
    f1, f2 = get_f((u1, u2), p)

    print("u_1 =", u1)
    print("u_2 =", u2)
    print("p =", p)

    print("\ngrad u_1 =", grad(u1))
    print("grad u_2 =", grad(u2))
    print("grad p =", grad(p))

    print("\nf_1 =", f1)
    print("f_2 =", f2)

    F1, F2 = get_f2(u)
    print("\nf_1 =", simplify(F1))
    print("f_2 =", simplify(F2))
