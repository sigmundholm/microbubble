#include <iomanip>
#include <iostream>
#include <vector>

#include "cutfem/geometry/SignedDistanceSphere.h"

#include "stokes.h"

#include "../stokes_time/stokes_gen.h"
#include "../projections/projection_flow.h"

template<int dim>
void solve_for_element_order(int element_order, int max_refinement,
                             bool write_output) {
    using namespace examples::cut::StokesEquation;
    using namespace examples::cut;

    double radius = 0.05;
    double half_length = radius;

    double nu = 1;

    double end_time = radius;

    double sphere_radius = 0.75 * radius;
    double sphere_x_coord = 0;

    std::ofstream file("errors-d" + std::to_string(dim)
                       + "o" + std::to_string(element_order) + ".csv");
    StokesEqn<dim>::write_header_to_file(file);

    RightHandSide<dim> rhs(nu);
    BoundaryValues<dim> boundary_values(nu);
    AnalyticalVelocity<dim> analytical_velocity(nu);
    AnalyticalPressure<dim> analytical_pressure(nu);
    MovingDomain<dim> domain(sphere_radius, half_length, radius);

    Point<dim> center;
    center = Point<dim>(sphere_x_coord,
                        0); // TODO trengs ny constructor for 3D?
    cutfem::geometry::SignedDistanceSphere<dim> signed_distance_sphere(
            sphere_radius, center, -1);

    for (int n_refines = 3; n_refines < max_refinement + 1; ++n_refines) {
        std::cout << "\nn_refines=" << n_refines << std::endl
                  << "===========" << std::endl;
        // Se feilen for tidsdiskretiseringen dominere hvis n_refines starter på
        // feks 3, og regn ut tau fra tau_init/2^(n_refines -3)
        double time_steps = pow(2, n_refines - 1);
        double tau = end_time / time_steps;


        std::cout << "T = " << end_time << ", tau = " << tau
                  << ", steps = " << time_steps << std::endl << std::endl;

        // L^2 projection of the step u0.
        analytical_velocity.set_time(0);
        analytical_pressure.set_time(0);
        /*
        projections::ProjectionFlow<dim> u0_proj(
                radius, half_length, n_refines, element_order, write_output,
                domain, analytical_velocity,
                analytical_pressure,
                sphere_radius, sphere_x_coord);
        ErrorBase err_proj = u0_proj.run_step();
        auto *error_proj = dynamic_cast<ErrorFlow*>(err_proj);
        std::cout << "  || u - u_h ||_L2 = " << error_proj.l2_error_u << std::endl;
        std::cout << "  || u - u_h ||_H1 = " << error_proj.h1_error_u << std::endl;
        std::cout << "  || p - p_h ||_L2 = " << error_proj.l2_error_p << std::endl;
        std::cout << "  || p - p_h ||_H1 = " << error_proj.h1_error_p << std::endl;
        Vector<double> u0 = u0_proj.get_solution();
         */

        StokesEqn<dim> stokes_bdf1(
                nu, tau, radius, half_length, n_refines, element_order,
                write_output, rhs, boundary_values, analytical_velocity,
                analytical_pressure, domain);

        ErrorBase *bdf1_err = stokes_bdf1.run_moving_domain(1, time_steps, 1.333);

        /*
        // std::cout << std::endl << "BDF-2" << std::endl << std::endl;
        StokesCylinder<dim> stokes_bdf2(
                radius, half_length, n_refines,
                nu, tau, element_order, write_output,
                rhs,
                boundary_values, analytical_velocity,
                analytical_pressure,
                sphere_radius, sphere_x_coord);

        Vector<double> u1 = stokes_bdf1.get_solution();
        std::vector<Vector<double>> initial2 = {u0, u1};
        TimeDependentStokesBDF2::Error error2 = stokes_bdf2.run(2, time_steps,
                                                                initial2);
         */
        auto *error = dynamic_cast<ErrorFlow *>(bdf1_err);

        std::cout << std::endl;
        std::cout << "|| u - u_h ||_L2 = " << error->l2_error_u << std::endl;
        std::cout << "|| u - u_h ||_H1 = " << error->h1_error_u << std::endl;
        std::cout << "|| p - p_h ||_L2 = " << error->l2_error_p << std::endl;
        std::cout << "|| p - p_h ||_H1 = " << error->h1_error_p << std::endl;
        StokesEqn<dim>::write_error_to_file(error, file);
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