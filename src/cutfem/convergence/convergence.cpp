#include <iomanip>
#include <iostream>
#include <vector>

#include "../error/ErrorStokesCylinder.h"

template<int dim>
void solve_for_element_order(int element_order, int max_refinement,
                             bool write_output) {

    double radius = 0.205;
    double half_length = 1.1;
    double pressure_drop = 10;

    std::ofstream file("errors-d" + std::to_string(dim)
                       + "o" + std::to_string(element_order) + ".csv");
    ErrorStokesCylinder<dim>::write_header_to_file(file);

    ErrorStokesRhs<dim> stokes_rhs(radius, 2 * half_length, pressure_drop);
    ErrorBoundaryValues<dim> boundary_values(radius, 2 * half_length,
                                             pressure_drop);

    for (int n_refines = 1; n_refines < max_refinement + 1; ++n_refines) {
        std::cout << "  n_refines=" << n_refines << std::endl;

        ErrorStokesCylinder<dim> s(radius, half_length, n_refines,
                                   element_order,
                                   write_output, stokes_rhs, boundary_values,
                                   pressure_drop);
        Error error = s.compute_error();
        ErrorStokesCylinder<dim>::write_error_to_file(error, file);

        std::cout << "    error=" << error.l2_error << std::endl;
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
    // TODO lag en base klasse for FEM-error, som har en run metode som
    //  returnerer double, og som har en konstruktør med refinements og element
    //  order

    run_convergence_test<2>({1}, 5, true);

}