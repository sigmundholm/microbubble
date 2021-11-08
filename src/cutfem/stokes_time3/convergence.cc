#include <iomanip>
#include <iostream>
#include <vector>

#include "stokes.h"

template<int dim>
void solve_for_element_order(int element_order, int max_refinement,
                             bool write_output) {
    using namespace examples::cut::StokesEquation2;

    double radius = 0.05;
    double half_length = radius;
    const double end_time = radius / 2;

    double delta = 1.3;
    double nu = 10;
    double tau;

    int time_steps;
    double sphere_radius = radius / 4;

    std::ofstream file("errors-d" + std::to_string(dim)
                       + "o" + std::to_string(element_order) + ".csv");
    StokesEqn2<dim>::write_header_to_file(file);

    RightHandSide<dim> rhs(delta, nu, tau);
    BoundaryValues<dim> boundary_values(nu);
    AnalyticalVelocity<dim> analytical_velocity(nu);
    AnalyticalPressure<dim> analytical_pressure(nu);
    MovingDomain<dim> domain(sphere_radius, half_length, radius);

    for (int n_refines = 1; n_refines < max_refinement + 1; ++n_refines) {
        std::cout << "\nn_refines=" << n_refines << std::endl;

        time_steps = pow(2, n_refines - 1);
        tau = end_time / time_steps;

        StokesEqn2<dim> stokes(radius, half_length, n_refines, delta, nu,
                                   tau, element_order, write_output, rhs,
                                   boundary_values, analytical_velocity,
                                   analytical_pressure, domain);
        ErrorBase *err = stokes.run_time(2, time_steps);
        auto *error = dynamic_cast<ErrorFlow *>(err);

        std::cout << "|| u - u_h ||_L2 = " << error->l2_error_u << std::endl;
        std::cout << "|| u - u_h ||_H1 = " << error->h1_error_u << std::endl;
        std::cout << "|| p - p_h ||_L2 = " << error->l2_error_p << std::endl;
        std::cout << "|| p - p_h ||_H1 = " << error->h1_error_p << std::endl;
        StokesEqn2<dim>::write_error_to_file(error, file);
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