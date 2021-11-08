from utils.expressions import *


def get_f(u, p):
    u1, u2 = u
    p_x, p_y = grad(p)
    f1 = sp.diff(u1, t) - nu * laplace(u1) + p_x
    f2 = sp.diff(u2, t) - nu * laplace(u2) + p_y
    return f1, f2


if __name__ == '__main__':
    """
    Here we want to solve the equations
        
        u - τνΔu + τ∇p = f
    """
    delta, nu, tau = sp.var("delta nu tau")
    t = sp.var("t")

    u1, u2 = get_u(t)
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
