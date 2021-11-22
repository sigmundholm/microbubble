#include "falling_sphere.h"

int main() {
    using namespace cut::fsi::falling_sphere;

    const int dim = 2;

    const unsigned int n_refines = 5;
    const unsigned int element_order = 1;
    const bool write_vtk = true;

    const double nu = 0.001;
    const double half_length = 1;
    const double radius = 2 * half_length;

    const double tau = 0.01;
    const unsigned int n_steps = 20;

    const double sphere_radius = 0.5 * radius;

    const double x0 = 0.1;
    const double y0 = 0;

    ZeroTensorFunction<1, dim> zero_tensor;
    Functions::ZeroFunction<dim> zero_scalar;
    RightHandSide<dim> rhs;
    BoundaryValues<dim> boundary(half_length, radius, sphere_radius);
    MovingDomain<dim> domain(half_length, radius, sphere_radius, x0, y0);

    FallingSphere<dim> ns(nu, tau, radius, half_length, n_refines,
                          element_order, write_vtk, rhs,
                          boundary, zero_tensor, zero_scalar,
                          domain, 1, 0.5, sphere_radius, true, 0);
    ns.run_moving_domain(1, n_steps);
    // TODO NB: this examples does not seem to work, the pressure field
}


