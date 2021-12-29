#include <iomanip>
#include <iostream>
#include <vector>

#include "poisson.h"


using namespace cutfem;
using namespace utils::problems::scalar;


template<int dim>
void solve_for_element_order(int element_order, int max_refinement,
                             bool write_output) {

    std::ofstream file("errors-d" + std::to_string(dim)
                       + "o" + std::to_string(element_order) + ".csv");

    double radius = 1.1;
    double half_length = 1.1;

    RightHandSide<dim> rhs;
    BoundaryValues<dim> bdd;
    AnalyticalSolution<dim> soln;

    // double sphere_radius = 1.0;
    FlowerDomain<dim> domain;

    for (int n_refines = 1; n_refines < max_refinement + 1; ++n_refines) {
        std::cout << "\nn_refines=" << n_refines << std::endl;

        Poisson<dim> poisson(radius, half_length, n_refines, element_order, write_output,
                             rhs, bdd, soln, domain);
        if (n_refines == 1) poisson.write_header_to_file(file);

        ErrorBase *err = poisson.run_step();
        auto *error = dynamic_cast<ErrorScalar*>(err);

        std::cout << "|| u - u_h ||_L2 = " << error->l2_error << std::endl;
        std::cout << "|| u - u_h ||_H1 = " << error->h1_error << std::endl;
        std::cout << "| u - u_h |_H1 = " << error->h1_semi << std::endl;
        poisson.write_error_to_file(err, file);
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


int main(int argc, char *argv[]) {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    run_convergence_test<2>({1, 2}, 8, false);
}