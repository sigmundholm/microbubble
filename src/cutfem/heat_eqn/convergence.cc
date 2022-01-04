#include <iomanip>
#include <iostream>
#include <vector>

#include "heat_eqn.h"


using namespace examples::cut::HeatEquation;


template<int dim>
void solve_for_element_order(int element_order, int max_refinement,
                             bool write_output) {

    std::ofstream file("errors-d" + std::to_string(dim)
                       + "o" + std::to_string(element_order) + ".csv");

    double radius = 1.0;
    double half_length = radius;

    const double nu = 2;
    const double end_time = 1;

    BoundaryValues<dim> bdd;
    AnalyticalSolution<dim> soln;

    double sphere_radius = 0.75 * radius;

    MovingDomain<dim> domain(sphere_radius, half_length, radius);
    // FlowerDomain<dim> domain;

    for (int n_refines = 3; n_refines < max_refinement + 1; ++n_refines) {
        std::cout << "\nn_refines=" << n_refines << std::endl;
        std::cout << "=========================" << std::endl;

        double time_steps = pow(2, n_refines - 1);
        // time_steps = 2;
        const double tau = end_time / time_steps;
        RightHandSide<dim> rhs(nu, tau);

        // BDF-1
        double bdf1_steps = pow(2, n_refines - 2);
        double bdf1_tau = tau / bdf1_steps;
        HeatEqn<dim> heat(nu, tau, radius, half_length, n_refines,
                          element_order, write_output,
                          rhs, bdd, soln, domain, true, false);
        ErrorBase *err = heat.run_time(1, 1);
        std::cout << std::endl;
        auto *error = dynamic_cast<ErrorScalar *>(err);

        std::cout << "|| u - u_h ||_L2 = " << error->l2_error << std::endl;
        std::cout << "|| u - u_h ||_H1 = " << error->h1_error << std::endl;
        std::cout << "| u - u_h |_H1 = " << error->h1_semi << std::endl;

        LA::MPI::Vector u1 = heat.get_solution();
        // std::shared_ptr<hp::DoFHandler<dim>> u1_dof_h = heat.get_dof_handler();

        // BDF-2
        std::vector<LA::MPI::Vector> initial = {u1};
        // std::vector<std::shared_ptr<hp::DoFHandler<dim>>> initial_dof_h = {
                // u1_dof_h};
        ErrorBase *err2 = heat.run_time(2, time_steps, initial); //, initial_dof_h);
        auto *error2 = dynamic_cast<ErrorScalar *>(err2);
        // auto *error2 = dynamic_cast<ErrorScalar*>(err);
        // TODO hvorfor er interpolasjonsfeilen mye høyre enn faktisk numerisk feil??

        std::cout << "|| u - u_h ||_L2 = " << error2->l2_error << std::endl;
        std::cout << "|| u - u_h ||_H1 = " << error2->h1_error << std::endl;
        std::cout << "| u - u_h |_H1 = " << error2->h1_semi << std::endl;

        if (n_refines == 3)
            heat.write_header_to_file(file);
        heat.write_error_to_file(error2, file);
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


int main(int argc, char *argv[]) {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    run_convergence_test<2>({1, 2}, 7, true);
}