#include "navier_stokes.h"

#include "../utils/flow_problem.h" // TODO må denne inkluderes?


int main() {
    using namespace examples::cut::NavierStokes;
    using namespace utils::problems::flow;

    const unsigned int n_refines = 4;
    const unsigned int elementOrder = 1;

    printf("numRefines=%d\n", n_refines);
    printf("elementOrder=%d\n", elementOrder);
    const bool write_vtk = true;
    const int dim = 2;

    const double radius = 1;
    const double half_length = 2 * radius;

    const double nu = 0.01;
    const double end_time = 1;
    unsigned int n_steps = 100;
    const double tau = end_time / n_steps;

    const double sphere_radius = radius * 0.3;

    RightHandSide<dim> rhs(nu);

    // BoundaryValues<dim> boundary(nu);
    const double max_speed = half_length;
    ParabolicFlow<dim> boundary(radius, half_length, max_speed, 0.1);

    AnalyticalVelocity<dim> analytical_velocity(nu);
    AnalyticalPressure<dim> analytical_pressure(nu);

    // MovingDomain<dim> domain(sphere_radius, half_length, radius);
    Sphere<dim> domain(sphere_radius, half_length, radius,
                       -half_length / 3, radius / 3);

    NavierStokesEqn<dim> ns(nu, tau, radius, half_length, n_refines,
                            elementOrder, write_vtk, rhs, boundary,
                            analytical_velocity, analytical_pressure,
                            domain, false, 2);

    ErrorBase *err = ns.run_time(1, n_steps);
    auto *error = dynamic_cast<ErrorFlow *>(err);

    std::cout << "Mesh size h = " << error->h << std::endl;
    std::cout << "|| u - u_h ||_L2 = " << error->l2_error_u << std::endl;
    std::cout << "|| u - u_h ||_H1 = " << error->h1_error_u << std::endl;
    std::cout << "| u - u_h |_H1 = " << error->h1_semi_u << std::endl;
    std::cout << "|| p - p_h ||_L2 = " << error->l2_error_p << std::endl;
    std::cout << "|| p - p_h ||_H1 = " << error->h1_error_p << std::endl;
    std::cout << "| p - p_h |_H1 = " << error->h1_semi_p << std::endl;
}
