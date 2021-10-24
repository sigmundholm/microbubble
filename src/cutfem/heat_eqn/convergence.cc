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

    double radius = 1;
    double half_length = radius;

    const double nu = 2;
    const double end_time = 1;

    BoundaryValues<dim> bdd;
    AnalyticalSolution<dim> soln;

    double sphere_radius = 0.75 * radius;
    double sphere_x_coord = 0;
    Point<dim> sphere_center;
    if (dim == 2) {
        sphere_center = Point<dim>(0, 0);
    } else if (dim == 3) {
        sphere_center = Point<dim>(0, 0, 0);
    }
    // cutfem::geometry::SignedDistanceSphere<dim> domain(sphere_radius, sphere_center, 1);

    MovingDomain<dim> domain(sphere_radius, half_length, radius);
    // FlowerDomain<dim> domain;

    for (int n_refines = 3; n_refines < max_refinement + 1; ++n_refines) {
        std::cout << "\nn_refines=" << n_refines << std::endl;
        std::cout << "=========================" << std::endl;

        double time_steps = pow(2, n_refines - 1);
        const double tau = end_time / time_steps;
        RightHandSide<dim> rhs(nu, tau);

        HeatEqn<dim> heat(nu, tau, radius, half_length, n_refines,
                          element_order, write_output,
                          rhs, bdd, soln, domain, true, false);

        // BDF-1
        heat.run_moving_domain(1, 1, 3);

        // BDF-2
        Vector<double> u1 = heat.get_solution();
        std::shared_ptr<hp::DoFHandler<dim>> u1_dof_h = heat.get_dof_handler();
        std::vector<Vector<double>> initial = {u1};
        std::vector<std::shared_ptr<hp::DoFHandler<dim>>> initial_dof_h =
                {u1_dof_h};

        heat.run_moving_domain(2, 2, initial, initial_dof_h, 1.5);

        // BDF-3
        Vector<double> u2 = heat.get_solution();
        std::shared_ptr<hp::DoFHandler<dim>> u2_dof_h = heat.get_dof_handler();
        std::vector<Vector<double>> initial2 = {u1, u2};
        std::vector<std::shared_ptr<hp::DoFHandler<dim>>> initial_dof_h2 =
                {u1_dof_h, u2_dof_h};

        ErrorBase *err3 = heat.run_moving_domain(3, time_steps, initial2,
                                                 initial_dof_h2, 1);
        auto *error = dynamic_cast<ErrorScalar *>(err3);

        std::cout << "|| u - u_h ||_L2 = " << error->l2_error << std::endl;
        std::cout << "|| u - u_h ||_H1 = " << error->h1_error << std::endl;
        std::cout << "| u - u_h |_H1 = " << error->h1_semi << std::endl;

        HeatEqn<dim>::write_error_to_file(error, file);
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
    run_convergence_test<2>({2, 3}, 7, true);
}