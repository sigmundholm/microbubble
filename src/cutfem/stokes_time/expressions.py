from utils.expressions import *


def get_f(u, p):
    u1, u2 = u
    p_x, p_y = grad(p)
    f1 = sp.diff(u1, t) - nu * laplace(u1) + p_x
    f2 = sp.diff(u2, t) - nu * laplace(u2) + p_y
    return f1, f2


def get_f_stokes_gen(u, p):
    u1, u2 = u
    p_x, p_y = grad(p)
    f1 = (delta * u1 - tau * nu * laplace(u1) + tau * p_x - u1) / tau
    f2 = (delta * u2 - tau * nu * laplace(u2) + tau * p_y - u2) / tau
    return f1, f2


def u_parabolic(r, t):
    """
    Equation for parabolic flow in pipe, weighted by sin(t)

    :param r: radius of pipe
    :param t:
    :return:
    """

    x, y = sp.var("x y")
    p = -(y - r) * (y + r)
    return p * sp.sin(t), 0


if __name__ == '__main__':
    """
    Here we want to solve the equations
        
        ∂_t u - νΔu + ∇p = f
    """
    t, nu, tau = sp.var("t nu tau")
    delta = sp.var('delta')
    r = sp.var('r')

    u1, u2 = u_parabolic(r, t)
    p = get_p(t)  # TODO må regnes ut ved å bruke ligningen og løse for p
    f1, f2 = 0, 0  # For parabolsk strøm

    print("u_1 =", u1)
    print("u_2 =", u2)
    print("p =", p)

    print("\ngrad u_1 =", grad(u1))
    print("grad u_2 =", grad(u2))
    print("grad p =", grad(p))

    print("\nf_1 =", f1)
    print("f_2 =", f2)
