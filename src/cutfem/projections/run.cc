#include "projection_flow.h"


int main() {
    using namespace GeneralizedStokes;

    const unsigned int n_refines = 4;
    const unsigned int elementOrder = 1;

    printf("numRefines=%d\n", n_refines);
    printf("elementOrder=%d\n", elementOrder);
    const bool write_vtk = true;

    double radius = 0.205;
    double half_length = 1.1;

    double delta = 1;
    double nu = 1;
    double tau = 1;

    double sphere_radius = radius / 4;
    double sphere_x_coord = -half_length / 2;

    const int dim = 2;

    RightHandSide<dim> rhs(delta, nu, tau);
    BoundaryValues<dim> boundary;
    AnalyticalVelocity<dim> analytical_velocity;
    AnalyticalPressure<dim> analytical_pressure;

    StokesCylinder<dim> stokes(radius, half_length, n_refines, delta, nu, tau,
                               elementOrder, write_vtk, rhs, boundary,
                               analytical_velocity, analytical_pressure,
                               sphere_radius, sphere_x_coord);
    Error error = stokes.run();
    std::cout << "Mesh size: " << error.mesh_size << std::endl;
    std::cout << "|| u - u_h ||_L2 = " << error.l2_error_u << std::endl;
    std::cout << "|| u - u_h ||_H1 = " << error.h1_error_u << std::endl;
    std::cout << "| u - u_h |_H1 = " << error.h1_semi_u << std::endl;
    std::cout << "|| p - p_h ||_L2 = " << error.l2_error_p << std::endl;
    std::cout << "|| p - p_h ||_H1 = " << error.h1_error_p << std::endl;
    std::cout << "| p - p_h |_H1 = " << error.h1_semi_p << std::endl;
}
