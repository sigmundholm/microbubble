#include <iostream>
#include <vector>

#include "cutfem/geometry/SignedDistanceSphere.h"

#include "../navier_stokes/navier_stokes.h"
#include "../navier_stokes/rhs_stat.h"

template<int dim>
void solve_for_element_order(int element_order, int max_refinement,
                             bool write_output) {
    using namespace examples::cut::NavierStokes;
    using namespace examples::cut;
    using namespace utils::problems::flow;

    double radius = 0.05;
    double half_length = radius;

    double nu = 0.1;

    double tau = 0;
    double end_time = radius;

    double sphere_radius = 0.75 * radius;
    double sphere_x_coord = 0;

    const bool semi_implicit = true;
    const int bdf_type = 2;

    std::ofstream file("errors-stat-d" + std::to_string(dim)
                       + "o" + std::to_string(element_order) + ".csv");
    std::ofstream meta("errors-stat-meta-d" + std::to_string(dim)
                       + "o" + std::to_string(element_order) + ".txt");
    meta << "Stationary Navier-Stokes convergence test" << std::endl
         << "==============================" << std::endl
         << "radius = " << radius << std::endl
         << "half_length = " << half_length << std::endl
         << "end_time = " << end_time << std::endl
         << "sphere_radius = " << sphere_radius << std::endl
         << "nu = " << nu << std::endl
         << "bdf_type = " << bdf_type << std::endl
         << "semi_implicit = " << semi_implicit << std::endl << std::endl;

    NavierStokesEqn<dim>::write_header_to_file(file);

    stationary::RightHandSide<dim> rhs(nu);
    stationary::ConvectionField<dim> conv_field(nu);
    stationary::AnalyticalVelocity<dim> analytical_velocity(nu);
    stationary::AnalyticalVelocity<dim> boundary_values(nu);
    stationary::AnalyticalPressure<dim> analytical_pressure(nu);
    MovingDomain<dim> domain(sphere_radius, half_length, radius);

    for (int n_refines = 3; n_refines < max_refinement + 1; ++n_refines) {
        std::cout << "\nn_refines=" << n_refines << std::endl
                  << "===========" << std::endl;
        meta << " - n_refines = " << n_refines << std::endl;

        double time_steps = pow(2, n_refines - 1);

        std::cout << "T = " << end_time << ", tau = " << tau
                  << ", steps = " << time_steps << std::endl << std::endl;

        NavierStokesEqn<dim> ns(nu, tau, radius, half_length, n_refines,
                                element_order, write_output, rhs, conv_field,
                                boundary_values,
                                analytical_velocity, analytical_pressure,
                                domain, semi_implicit, 10, true,
                                true);

        ErrorBase *err = ns.run_step_non_linear(1e-10);
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
