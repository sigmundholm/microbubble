#include "ex_falling_sphere.h"

int main() {
    using namespace examples::cut::StokesEquation;

    const int dim = 2;

    const unsigned int n_refines = 4;
    const unsigned int element_order = 1;
    const bool write_vtk = true;

    const double radius = 2;
    const double half_length = 0.5 * radius;
    const double nu = 1;
    const double tau = 0.01;
    const unsigned int n_steps = 100;

    const double sphere_radius = half_length * 0.75;


    ZeroTensorFunction<1, dim> zero_tensor;
    Functions::ZeroFunction<dim> zero_scalar;
    ex2::BoundaryValues<dim> boundary(sphere_radius, half_length, radius);
    ex2::MovingDomain<dim> domain(sphere_radius, half_length, radius, tau);

    ex2::FallingSphereStokes<dim> stokes(nu, tau, radius, half_length, n_refines,
                                   element_order, write_vtk, zero_tensor,
                                   boundary,
                                   zero_tensor, zero_scalar, domain);
    stokes.run_moving_domain(1, 20);


}


