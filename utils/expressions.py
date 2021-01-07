import sympy as sp
from sympy import simplify


def u_func(x, y, eps, pck=sp):
    return pck.cos(pck.pi) * (1 - pck.exp((y - 1) / eps)) / (
            1 - pck.exp(-2 / eps)) + 0.5 * pck.cos(pck.pi * x) \
           * pck.sin(pck.pi * y)


def grad(func):
    x, y = sp.var("x y")
    return sp.diff(func, x), sp.diff(func, y)


def div(func):
    x, y = sp.var("x y")
    return sp.diff(func[0], x) + sp.diff(func[1], y)


def laplace(func):
    x, y = sp.var("x y")
    return sp.diff(sp.diff(func, x), x) + sp.diff(sp.diff(func, y), y)


def get_u(t=0):
    """Ethier-Steinman (1994)"""
    x, y, nu = sp.var("x y nu")
    u1 = -sp.cos(sp.pi * x) * sp.sin(sp.pi * y) * sp.exp(-2 * sp.pi ** 2 * nu * t)
    u2 = sp.sin(sp.pi * x) * sp.cos(sp.pi * y) * sp.exp(-2 * sp.pi ** 2 * nu * t)
    return u1, u2


def get_p(t=0):
    """Ethier-Steinman (1994)"""
    x, y, nu = sp.var("x y nu")
    return -(sp.cos(2 * sp.pi * x) + sp.cos(2 * sp.pi * y)) / 4 * sp.exp(-4 * sp.pi ** 2 * nu * t)


def convection(u):
    """
    Calculate the convection term: (u·∇)u
    """
    x, y, nu = sp.var("x y nu")
    u1, u2 = u
    return u1 * sp.diff(u1, x) + u2 * sp.diff(u1, y), \
           u1 * sp.diff(u2, x) + u2 * sp.diff(u2, y)


def convection_term(b, u):
    """
    Calculate a convection term: b·∇u.
    Should be identical to the result of `convection` above when b = u.
    """
    b1, b2 = b
    u1, u2 = u
    grad_u1 = grad(u1)
    grad_u2 = grad(u2)

    return (b1 * grad_u1[0] + b2 * grad_u1[1],
            b1 * grad_u2[0] + b2 * grad_u2[1])


if __name__ == '__main__':
    def check(u, p):
        x, y, t, nu = sp.var("x y t nu")
        u1, u2 = u
        px, py = grad(p)
        c1, c2 = convection(u)

        # Navier-Stokes equations
        f1 = sp.diff(u1, t) + c1 - nu * laplace(u1) + px
        f2 = sp.diff(u2, t) + c2 - nu * laplace(u2) + py

        print("\nCheck that u and p solves the homogeneous NS:")
        print("f1 =", simplify(f1))
        print("f2 =", simplify(f2))
        print("div u =", div(u))


    t = sp.var("t")
    u = get_u(t)
    p = get_p(t)
    check(u, p)
