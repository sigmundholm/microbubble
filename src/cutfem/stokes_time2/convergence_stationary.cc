#include <iostream>
#include <vector>

#include "cutfem/geometry/SignedDistanceSphere.h"

#include "../stokes_time2/stokes.h"
#include "../stokes_time2/rhs_stat.h"


template<int dim>
void solve_for_element_order(int element_order, int max_refinement,
                             bool write_output) {
    using namespace examples::cut::StokesEquation;
    using namespace examples::cut;
    using namespace utils::problems::flow;

    double radius = 1;
    double half_length = radius;

    double nu = 2;

    double tau = 1;

    double sphere_radius = 0.75 * radius;
    double sphere_x_coord = 0;

    std::ofstream file("errors-stat-d" + std::to_string(dim)
                       + "o" + std::to_string(element_order) + ".csv");

    std::ofstream file_stresses("e-flux-d" + std::to_string(dim)
                                + "o" + std::to_string(element_order) + ".csv");
    file_stresses << "h; exact; regular; nitsche; symmetric; sym_nitsche"
                  << std::endl;

    std::ofstream meta("errors-stat-meta-d" + std::to_string(dim)
                       + "o" + std::to_string(element_order) + ".txt");
    meta << "Stationary Stokes convergence test" << std::endl
         << "==================================" << std::endl
         << "radius = " << radius << std::endl
         << "half_length = " << half_length << std::endl
         << "sphere_radius = " << sphere_radius << std::endl
         << "nu = " << nu << std::endl;

    StokesEqn<dim>::write_header_to_file(file);

    stationary::RightHandSide<dim> rhs(nu);
    AnalyticalVelocity<dim> analytical_velocity(nu);
    AnalyticalVelocity<dim> boundary_values(nu);
    AnalyticalPressure<dim> analytical_pressure(nu);
    MovingDomain<dim> domain(sphere_radius, half_length, radius);

    for (int n_refines = 3; n_refines < max_refinement + 1; ++n_refines) {
        std::cout << "\nn_refines=" << n_refines << std::endl
                  << "-----------" << std::endl;
        meta << " - n_refines = " << n_refines << std::endl;

        double n_cells = pow(2, n_refines - 1);
        double h = radius / n_cells;


        StokesEqn<dim> stokes(nu, tau, radius, half_length, n_refines,
                              element_order, write_output, rhs,
                              boundary_values,
                              analytical_velocity, analytical_pressure,
                              domain, 10, true, true,
                              true);

        ErrorBase *err = stokes.run_step();
        auto *error = dynamic_cast<ErrorFlow *>(err);

        std::cout << std::endl;
        std::cout << "|| u - u_h ||_L2 = " << error->l2_error_u << std::endl;
        std::cout << "|| u - u_h ||_H1 = " << error->h1_error_u << std::endl;
        std::cout << "|| p - p_h ||_L2 = " << error->l2_error_p << std::endl;
        std::cout << "|| p - p_h ||_H1 = " << error->h1_error_p << std::endl;
        StokesEqn<dim>::write_error_to_file(error, file);

        // Compute the stress forces on the sphere, using the
        // different approaches.
        double nitsche = stokes.compute_surface_forces(
                Stress::NitscheFlux | Stress::Error)[0];
        double exact = stokes.compute_surface_forces(
                Stress::Exact | Stress::Error)[0];
        double regular = stokes.compute_surface_forces(
                Stress::Regular | Stress::Error)[0];
        double symmetric = stokes.compute_surface_forces(
                Stress::Symmetric | Stress::Error)[0];
        double sym_nitsche = stokes.compute_surface_forces(
                Stress::NitscheFlux | Stress::Symmetric | Stress::Error)[0];

        file_stresses << h << ";"
                      << exact << ";" << regular << ";" << nitsche << ";"
                      << symmetric << ";" << sym_nitsche << std::endl;
    }
}


template<int dim>
void run_convergence_test(std::vector<int> orders, int max_refinement,
                          bool write_output) {
    for (int order : orders) {
        std::cout << "\ndim=" << dim << ", element_order=" << order << std::endl;
        std::cout << "====================" << std::endl;
        solve_for_element_order<dim>(order, max_refinement, write_output);
    }
}


int main() {
    run_convergence_test<2>({1, 2}, 7, true);
}
