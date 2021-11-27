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
    double tau = 1;

    double sphere_radius = 0.75 * radius;
    double sphere_x_coord = 0;

    const bool semi_implicit = true;

    std::ofstream file_stresses("e-stress-d" + std::to_string(dim)
                                + "o" + std::to_string(element_order) + ".csv");
    file_stresses << "h; drag_exact; lift_exact; drag_regular; lift_regular; "
                     "drag_symmetric; lift_symmetric; "
                     "drag_nitsche; lift_nitsche" << std::endl;

    std::ofstream file_errors("errors-stat-d" + std::to_string(dim)
                              + "o" + std::to_string(element_order) + ".csv");
    NavierStokesEqn<dim>::write_header_to_file(file_errors);

    stationary::RightHandSide<dim> rhs(nu);
    stationary::AnalyticalVelocity<dim> analytical_velocity(nu);
    stationary::AnalyticalVelocity<dim> boundary_values(nu);
    stationary::AnalyticalPressure<dim> analytical_pressure(nu);
    MovingDomain<dim> domain(sphere_radius, half_length, radius);

    for (int n_refines = 3; n_refines < max_refinement + 1; ++n_refines) {
        std::cout << "\nn_refines=" << n_refines << std::endl
                  << "===========" << std::endl;
        // meta << " - n_refines = " << n_refines << std::endl;

        double grid_size = pow(2, n_refines - 1);
        double h = radius / grid_size;

        NavierStokesEqn<dim> ns(nu, tau, radius, half_length, n_refines,
                                element_order, write_output, rhs,
                                boundary_values,
                                analytical_velocity, analytical_pressure,
                                domain, semi_implicit, 10,
                                true, true);

        ErrorBase *err = ns.run_step_non_linear(1e-11);
        auto *error = dynamic_cast<ErrorFlow *>(err);
        NavierStokesEqn<dim>::write_error_to_file(error, file_errors);

        std::cout << std::endl;
        std::cout << "|| u - u_h ||_L2 = " << error->l2_error_u << std::endl;
        std::cout << "|| u - u_h ||_H1 = " << error->h1_error_u << std::endl;
        std::cout << "|| p - p_h ||_L2 = " << error->l2_error_p << std::endl;
        std::cout << "|| p - p_h ||_H1 = " << error->h1_error_p << std::endl;

        // Compute the stress forces on the sphere, using the
        // different approaches.
        Tensor<1, dim> exact = ns.compute_surface_forces(
                Stress::Exact | Stress::Test);
        Tensor<1, dim> regular = ns.compute_surface_forces(
                Stress::Regular | Stress::Test);
        Tensor<1, dim> symmetric = ns.compute_surface_forces(
                Stress::Symmetric | Stress::Test);
        Tensor<1, dim> nitsche = ns.compute_surface_forces(
                Stress::NitscheFlux | Stress::Test);

        regular -= exact;
        symmetric -= exact;
        nitsche -= exact;

        file_stresses << h << ";"
                      << exact[0] << ";" << exact[1] << ";"
                      << regular[0] << ";" << regular[1] << ";"
                      << symmetric[0] << ";" << symmetric[1] << ";"
                      << nitsche[0] << ";" << nitsche[1] << std::endl;

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
    std::cout << "\n\n=========end" << std::endl;
}
