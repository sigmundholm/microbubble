#include <iomanip>
#include <iostream>
#include <vector>

#include "poisson.h"


template<int dim>
void solve_for_element_order(int element_order, int max_refinement,
                             bool write_output) {

    std::ofstream file("errors-d" + std::to_string(dim)
                       + "o" + std::to_string(element_order) + ".csv");
    Poisson<dim>::write_header_to_file(file);

    double radius = 1.1;
    double half_length = 1.1;
    double sphere_rad = 1.0;
    double sphere_x_coord = 0;

    RightHandSide<dim> rhs;
    BoundaryValues<dim> bdd;
    AnalyticalSolution<dim> soln;

    for (int n_refines = 1; n_refines < max_refinement + 1; ++n_refines) {
        std::cout << "\nn_refines=" << n_refines << std::endl;


        Poisson<dim> poisson(radius, half_length, n_refines, element_order, write_output,
                             rhs, bdd, soln, sphere_rad, sphere_x_coord);
        Error error = poisson.run(false);

        std::cout << "|| u - u_h ||_L2 = " << error.l2_error << std::endl;
        std::cout << "|| u - u_h ||_H1 = " << error.h1_error << std::endl;
        std::cout << "| u - u_h |_H1 = " << error.h1_semi << std::endl;
        Poisson<dim>::write_error_to_file(error, file);
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