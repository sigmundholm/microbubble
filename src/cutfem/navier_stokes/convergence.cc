#include <iostream>
#include <vector>

#include "cutfem/geometry/SignedDistanceSphere.h"

#include "../navier_stokes/navier_stokes.h"

template<int dim>
void solve_for_element_order(int element_order, int max_refinement,
                             bool write_output) {
    using namespace examples::cut::NavierStokes;
    using namespace examples::cut;
    using namespace utils::problems::flow;

    double radius = 0.05;
    double half_length = radius;

    double nu = 10;

    double end_time = radius;

    double sphere_radius = 0.75 * radius;
    double sphere_x_coord = 0;

    std::ofstream file("errors-d" + std::to_string(dim)
                       + "o" + std::to_string(element_order) + ".csv");
    NavierStokesEqn<dim>::write_header_to_file(file);

    RightHandSide <dim> rhs;
    BoundaryValues <dim> boundary_values(nu);
    AnalyticalVelocity <dim> analytical_velocity(nu);
    AnalyticalPressure <dim> analytical_pressure(nu);
    MovingDomain <dim> domain(sphere_radius, half_length, radius);

    for (int n_refines = 3; n_refines < max_refinement + 1; ++n_refines) {
        std::cout << "\nn_refines=" << n_refines << std::endl
                  << "===========" << std::endl;

        double time_steps = pow(2, n_refines - 1);
        double tau = end_time / time_steps;

        std::cout << "T = " << end_time << ", tau = " << tau
                  << ", steps = " << time_steps << std::endl << std::endl;

        NavierStokesEqn <dim> ns(nu, tau, radius, half_length, n_refines,
                                 element_order, write_output, rhs,
                                 boundary_values,
                                 analytical_velocity, analytical_pressure,
                                 domain, false);

        ErrorBase *err = ns.run_time(2, time_steps);
        auto *error = dynamic_cast<ErrorFlow *>(err);

        std::cout << std::endl;
        std::cout << "|| u - u_h ||_L2 = " << error->l2_error_u << std::endl;
        std::cout << "|| u - u_h ||_H1 = " << error->h1_error_u << std::endl;
        std::cout << "|| p - p_h ||_L2 = " << error->l2_error_p << std::endl;
        std::cout << "|| p - p_h ||_H1 = " << error->h1_error_p << std::endl;
        NavierStokesEqn<dim>::write_error_to_file(error, file);
    }
}


template<int dim>
void run_convergence_test(std::vector<int> orders, int max_refinement,
                          bool write_output) {
    for (int order : orders) {
        std::cout << "dim=" << dim << ", element_order=" << order << std::endl;
        solve_for_element_order<dim>(order, max_refinement, write_output);
    }
}


int main() {
    run_convergence_test<2>({1, 2}, 7, true);
}
