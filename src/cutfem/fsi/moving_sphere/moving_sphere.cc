#include "rhs.h"

#include "../../navier_stokes/navier_stokes.h"

int main() {
    using namespace cut::fsi::moving_sphere;
    using namespace examples::cut;

    const int dim = 2;

    const int n_refines = 5;
    const int element_order = 1;
    const bool write_vtk = true;

    const double nu = 0.001;
    const double half_length = 2;
    const double radius = 1;
    const double sphere_radius = 0.5;

    const double end_time = 16;
    const unsigned int n_steps = 320;
    const double tau = end_time / n_steps;

    MovingDomain<dim> domain(half_length, radius, sphere_radius, radius / 5);

    BoundaryValues<dim> boundary(half_length, radius, sphere_radius, domain);


    ZeroTensorFunction<1, dim> zero_tensor;
    Functions::ZeroFunction<dim> zero_scalar;

    NavierStokes::NavierStokesEqn<dim> ns(
            nu, tau, radius, half_length, n_refines, element_order, write_vtk,
            zero_tensor, boundary, zero_tensor, zero_scalar,
            domain, true, 2, true, false, true);

    // BDF-1
    ns.run_moving_domain(1, 1);

    Vector<double> u1 = ns.get_solution();
    std::vector<Vector<double>> initial = {u1};
    std::shared_ptr<hp::DoFHandler<dim>> u1_dof = ns.get_dof_handler();
    std::vector<std::shared_ptr<hp::DoFHandler<dim>>> initial_dofs = {u1_dof};

    // BDF-2
    ns.run_moving_domain(2, n_steps, initial, initial_dofs);
}

