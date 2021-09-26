#include <iomanip>
#include <iostream>
#include <vector>

#include "heat_eqn.h"

#include "cutfem/geometry/SignedDistanceSphere.h"

using namespace cutfem;

using namespace examples::cut::HeatEquation;


template<int dim>
void solve_for_element_order(int element_order, int max_refinement,
                             bool write_output) {

    std::ofstream file("errors-d" + std::to_string(dim)
                       + "o" + std::to_string(element_order) + ".csv");
    HeatEqn<dim>::write_header_to_file(file);

    double radius = 1.1;
    double half_length = 1.1;

    const double nu = 2;
    const double end_time = 1.1;

    BoundaryValues<dim> bdd;
    AnalyticalSolution<dim> soln;

    double sphere_radius = 1.0;
    double sphere_x_coord = 0;
    Point<dim> sphere_center;
    if (dim == 2) {
        sphere_center = Point<dim>(0, 0);
    } else if (dim == 3) {
        sphere_center = Point<dim>(0, 0, 0);
    }
    cutfem::geometry::SignedDistanceSphere<dim> domain(sphere_radius, sphere_center, 1);
    // FlowerDomain<dim> domain;

    for (int n_refines = 2; n_refines < max_refinement + 1; ++n_refines) {
        std::cout << "\nn_refines=" << n_refines << std::endl;
        std::cout << "=========================" << std::endl;

        double time_steps = pow(2, n_refines - 1);
        // time_steps = 2;
        const double tau = end_time / time_steps;
        RightHandSide<dim> rhs(nu, tau);

        // BDF-1
        HeatEqn<dim> heat(nu, tau, radius, half_length, n_refines, element_order,
                             write_output,
                             rhs, bdd, soln, domain, true, false);
        ErrorBase *err = heat.run_time(1, 1);
        auto *error = dynamic_cast<ErrorScalar*>(err);

        std::cout << "|| u - u_h ||_L2 = " << error->l2_error << std::endl;
        std::cout << "|| u - u_h ||_H1 = " << error->h1_error << std::endl;
        std::cout << "| u - u_h |_H1 = " << error->h1_semi << std::endl;

        Vector<double> u1 = heat.get_solution();

        // BDF-2
        HeatEqn<dim> heat2(nu, tau, radius, half_length, n_refines, element_order,
                           write_output,
                           rhs, bdd, soln, domain, true, false);

        std::vector<Vector<double>> initial = {u1};
        ErrorBase *err2 = heat2.run_time(2, time_steps, initial);
        auto *error2 = dynamic_cast<ErrorScalar*>(err);

        std::cout << "|| u - u_h ||_L2 = " << error2->l2_error << std::endl;
        std::cout << "|| u - u_h ||_H1 = " << error2->h1_error << std::endl;
        std::cout << "| u - u_h |_H1 = " << error2->h1_semi << std::endl;

        HeatEqn<dim>::write_error_to_file(error2, file);
    }
}


template<int dim>
void run_convergence_test(std::vector<int> orders, int max_refinement,
                          bool write_output) {
    for (int order : orders) {
        std::cout << "dim=" << dim << ", element_order=" << order
                  << std::endl;
        solve_for_element_order<dim>(order, max_refinement, write_output);
    }
}


int main() {
    run_convergence_test<2>({1, 2}, 7, true);
}