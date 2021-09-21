#include <iomanip>
#include <iostream>
#include <vector>

#include "cutfem/geometry/SignedDistanceSphere.h"

#include "projection_flow.h"

using namespace examples::cut::projections;
using namespace cutfem;


template<int dim>
void solve_for_element_order(int element_order, int max_refinement,
                             bool write_output) {

    double radius = 0.5;
    double half_length = radius;

    double sphere_radius = 0.9 * radius;
    double sphere_x_coord = 0;

    std::ofstream file("errors-d" + std::to_string(dim)
                       + "o" + std::to_string(element_order) + ".csv");
    ProjectionFlow<dim>::write_header_to_file(file);

    AnalyticalVelocity<dim> u_0;
    AnalyticalPressure<dim> p_0;

    Point<dim> center;
    center = Point<dim>(sphere_x_coord, 0); // TODO trengs ny constructor for 3D?
    cutfem::geometry::SignedDistanceSphere<dim> signed_distance_sphere(
            sphere_radius, center, -1);

    for (int n_refines = 1; n_refines < max_refinement + 1; ++n_refines) {
        std::cout << "\nn_refines=" << n_refines << std::endl;

        ProjectionFlow<dim> stokes(radius, half_length, n_refines,
                                   element_order, write_output,
                                   signed_distance_sphere, u_0, p_0,
                                   sphere_radius, sphere_x_coord);
        Error error = stokes.run_step();

        std::cout << "|| u - u_h ||_L2 = " << error.l2_error_u << std::endl;
        std::cout << "|| u - u_h ||_H1 = " << error.h1_error_u << std::endl;
        std::cout << "|| p - p_h ||_L2 = " << error.l2_error_p << std::endl;
        std::cout << "|| p - p_h ||_H1 = " << error.h1_error_p << std::endl;
        ProjectionFlow<dim>::write_error_to_file(error, file);
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