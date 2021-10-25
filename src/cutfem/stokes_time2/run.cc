#include "stokes.h"


int main() {
    using namespace examples::cut::StokesEquation;

    const unsigned int n_refines = 4;
    const unsigned int elementOrder = 1;

    printf("numRefines=%d\n", n_refines);
    printf("elementOrder=%d\n", elementOrder);
    const bool write_vtk = true;

    double radius = 0.05;
    double half_length = 2 * radius;

    double nu = 1;
    double end_time = 0.05;
    unsigned int n_steps = 20;
    double tau = end_time / n_steps;

    double sphere_radius = radius * 0.75;

    const int dim = 2;

    RightHandSide<dim> rhs(nu);
    BoundaryValues<dim> boundary(nu);
    AnalyticalVelocity<dim> analytical_velocity(nu);
    AnalyticalPressure<dim> analytical_pressure(nu);
    MovingDomain<dim> domain(sphere_radius, half_length, radius);

    Point<dim> center = Point<dim>(0, 0);
    // cutfem::geometry::SignedDistanceSphere<dim> domain(sphere_radius, center, -1);

    StokesEqn<dim> stokes(nu, tau, radius, half_length, n_refines,
                               elementOrder, write_vtk, rhs, boundary,
                               analytical_velocity, analytical_pressure,
                               domain);

    ErrorBase *err = stokes.run_moving_domain(1, n_steps, 1.333);
    auto *error = dynamic_cast<ErrorFlow*>(err);

    std::cout << "Mesh size h = " << error->h << std::endl;
    std::cout << "|| u - u_h ||_L2 = " << error->l2_error_u << std::endl;
    std::cout << "|| u - u_h ||_H1 = " << error->h1_error_u << std::endl;
    std::cout << "| u - u_h |_H1 = " << error->h1_semi_u << std::endl;
    std::cout << "|| p - p_h ||_L2 = " << error->l2_error_p << std::endl;
    std::cout << "|| p - p_h ||_H1 = " << error->h1_error_p << std::endl;
    std::cout << "| p - p_h |_H1 = " << error->h1_semi_p << std::endl;
}
