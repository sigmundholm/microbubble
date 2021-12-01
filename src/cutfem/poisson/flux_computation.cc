#include <iostream>
#include <vector>

#include "cutfem/geometry/SignedDistanceSphere.h"

#include "poisson.h"
#include "rhs.h"


template<int dim>
void solve_for_element_order(int element_order, int max_refinement,
                             bool write_output) {
    using namespace cut::PoissonProblem;
    using namespace utils::problems::scalar;

    const double radius = 1;
    const double half_length = radius;
    const double sphere_radius = 0.9 * radius;

    const double nu = 10;
    double h;

    std::ofstream file_stresses("e-flux-d" + std::to_string(dim)
                                + "o" + std::to_string(element_order) + ".csv");
    file_stresses << "h; exact; regular; nitsche" << std::endl;

    std::ofstream file_errors("errors-d" + std::to_string(dim)
                              + "o" + std::to_string(element_order) + ".csv");

    Poisson<dim>::write_header_to_file(file_errors);

    RightHandSide <dim> rhs(nu);
    AnalyticalSolution <dim> solution;
    BoundaryValues <dim> boundary;

    Point <dim> sphere_center;
    if (dim == 2) {
        sphere_center = Point<dim>(0, 0);
    } else if (dim == 3) {
        sphere_center = Point<dim>(0, 0, 0);
    }
    cutfem::geometry::SignedDistanceSphere<dim> domain(sphere_radius, sphere_center, 1);
    // FlowerDomain <dim> domain;

    for (int n_refines = 3; n_refines < max_refinement + 1; ++n_refines) {
        h = radius / pow(2, n_refines - 1);

        std::cout << "\nn_refines=" << n_refines << std::endl
                  << "===========" << std::endl;

        Poisson <dim> poisson(nu, radius, half_length, n_refines, element_order,
                              write_output, rhs, boundary, solution, domain);

        ErrorBase *err = poisson.run_step();
        auto *error = dynamic_cast<ErrorScalar *>(err);
        Poisson<dim>::write_error_to_file(error, file_errors);

        std::cout << std::endl;
        std::cout << "|| u - u_h ||_L2 = " << error->l2_error << std::endl;
        std::cout << "|| u - u_h ||_H1 = " << error->h1_error << std::endl;

        // Compute the stress forces on the sphere, using the
        // different approaches.
        double exact = poisson.compute_surface_flux(
                Flux::Exact | Flux::Error);
        double regular = poisson.compute_surface_flux(
                Flux::Regular | Flux::Error);
        double nitsche = poisson.compute_surface_flux(
                Flux::NitscheFlux | Flux::Error);

        file_stresses << h << ";"
                      << exact << ";" << regular << ";"
                      << nitsche << std::endl;
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
    run_convergence_test<2>({1, 2}, 8, true);
}
