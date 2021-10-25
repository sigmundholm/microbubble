#include "cutfem/geometry/SignedDistanceSphere.h"

#include "projection_flow.h"

using namespace examples::cut::projections;
using namespace utils::problems::flow;
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

    const int dim = 2;

    AnalyticalVelocity<dim> u_0;
    AnalyticalPressure<dim> p_0;
    MovingDomain<dim> domain(sphere_radius, half_length, radius);

    ProjectionFlow<dim> proj(radius, half_length, n_refines,
                             elementOrder, write_vtk,
                             domain, u_0, p_0);
    ErrorBase *err = proj.run_step();
    auto *error = dynamic_cast<ErrorFlow *>(err);

    std::cout << "Mesh size: " << error->h << std::endl;
    std::cout << "|| u - u_h ||_L2 = " << error->l2_error_u << std::endl;
    std::cout << "|| u - u_h ||_H1 = " << error->h1_error_u << std::endl;
    std::cout << "| u - u_h |_H1 = " << error->h1_semi_u << std::endl;
    std::cout << "|| p - p_h ||_L2 = " << error->l2_error_p << std::endl;
    std::cout << "|| p - p_h ||_H1 = " << error->h1_error_p << std::endl;
    std::cout << "| p - p_h |_H1 = " << error->h1_semi_p << std::endl;
}
