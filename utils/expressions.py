import sympy as sp


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
