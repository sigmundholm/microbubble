import sympy as sp
from sympy.integrals import integrate

x1, x2, rho, r, m = sp.var("x1 x2 rho, r, m")

"""
Integrate over a circle of radius r, using polar coordinates.
"""
def integrate_circle(integrand):
    r_tilde, theta = sp.var("r_tilde, theta")
    integrand = integrand.subs(x1, r_tilde * sp.cos(theta))
    integrand = integrand.subs(x2, r_tilde * sp.sin(theta))

    tmp = integrate(integrand, (theta, 0, 2 * sp.pi))
    return integrate(tmp * r_tilde, (r_tilde, 0, r)).subs(rho, m / (sp.pi * r ** 2))


if __name__ == '__main__':
    # Compute the inertia tensor for the sphere in 2D.
    res = integrate_circle(rho * x2 ** 2)
    print("I_11 = ", res)

    res2 = integrate_circle(rho * x1 ** 2)
    print("I_22 = ", res2)

    res3 = integrate_circle(rho * (x1 ** 2 + x2 ** 2))
    print("I_33 = ", res3)

    res_12 = integrate_circle(rho * x1 * x2)
    print("I_12 =", res_12)

