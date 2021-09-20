#include "cutfem/geometry/SignedDistanceSphere.h"

#include "projection_flow.h"

using namespace examples::cut::projections;
using namespace cutfem;

int main() {
    const unsigned int n_refines = 4;
    const unsigned int elementOrder = 1;

    printf("numRefines=%d\n", n_refines);
    printf("elementOrder=%d\n", elementOrder);
    const bool write_vtk = true;

    double radius = 0.5;
    double half_length = radius;

    double sphere_radius = 0.9 * radius;
    double sphere_x_coord = 0;

    const int dim = 2;

    AnalyticalVelocity<dim> u_0;
    AnalyticalPressure<dim> p_0;

    Point<dim> center;
    center = Point<dim>(sphere_x_coord,
                        0); // TODO trengs ny constructor for 3D?
    cutfem::geometry::SignedDistanceSphere<dim> signed_distance_sphere(
            sphere_radius, center, -1);

    ProjectionFlow<dim> stokes(radius, half_length, n_refines,
                               elementOrder, write_vtk,
                               signed_distance_sphere, u_0, p_0,
                               sphere_radius, sphere_x_coord);
    Error error = stokes.run_step();
    std::cout << "Mesh size: " << error.h << std::endl;
    std::cout << "|| u - u_h ||_L2 = " << error.l2_error_u << std::endl;
    std::cout << "|| u - u_h ||_H1 = " << error.h1_error_u << std::endl;
    std::cout << "| u - u_h |_H1 = " << error.h1_semi_u << std::endl;
    std::cout << "|| p - p_h ||_L2 = " << error.l2_error_p << std::endl;
    std::cout << "|| p - p_h ||_H1 = " << error.h1_error_p << std::endl;
    std::cout << "| p - p_h |_H1 = " << error.h1_semi_p << std::endl;
}
