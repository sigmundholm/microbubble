#include <iomanip>
#include <iostream>
#include <vector>

#include "poisson.h"


template<int dim>
void solve_for_element_order(int element_order, int max_refinement,
                             bool write_output) {

    std::ofstream file("errors-d" + std::to_string(dim)
                       + "o" + std::to_string(element_order) + ".csv");
    PoissonNitsche<dim>::write_header_to_file(file);

    RightHandSide<dim> rhs;
    BoundaryValues<dim> boundary_values;
    AnalyticalSolution<dim> analytical_solution;

    for (int n_refines = 1; n_refines < max_refinement + 1; ++n_refines) {
        std::cout << "\nn_refines=" << n_refines << std::endl;

        PoissonNitsche<dim> poisson(element_order, n_refines);
        Error error = poisson.run();

        std::cout << "|| u - u_h ||_L2 = " << error.l2_error << std::endl;
        std::cout << "|| u - u_h ||_H1 = " << error.h1_error << std::endl;
        std::cout << "| u - u_h |_H1 = " << error.h1_semi << std::endl;
        PoissonNitsche<dim>::write_error_to_file(error, file);
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