#include "poisson.h"
#include "rhs.h"

#include "cutfem/geometry/SignedDistanceSphere.h"

using namespace cutfem;

using namespace utils::problems::scalar;

int main() {
    const int dim = 2;

    const double nu = 2;
    const double radius = 1;
    const double half_length = 1;
    const int n_refines = 5;
    const int degree = 1;
    const bool write_output = true;

    using namespace cut::PoissonProblem;

    RightHandSide<dim> rhs(nu);
    BoundaryValues<dim> bdd;
    AnalyticalSolution<dim> soln;

    double sphere_radius = radius * 0.9;
    double sphere_x_coord = 0;
    Point<dim> sphere_center;
    if (dim == 2) {
        sphere_center = Point<dim>(0, 0);
    } else if (dim == 3) {
        sphere_center = Point<dim>(0, 0, 0);
    }
    // cutfem::geometry::SignedDistanceSphere<dim> domain(sphere_radius, sphere_center, 1);

    FlowerDomain<dim> domain;

    Poisson<dim> poisson(nu, radius, half_length, n_refines, degree,
                         write_output, rhs, bdd, soln, domain);

    ErrorBase *err = poisson.run_step();
    auto *error = dynamic_cast<ErrorScalar *>(err);
    error->output();

    double regular = poisson.compute_surface_flux(Flux::Regular);
    double nitsche = poisson.compute_surface_flux(Flux::NitscheFlux);
    std::cout << "Boundary flux:"
              << "\n- Regular = " << regular
              << "\n- Nitsche flux = " << nitsche
              << std::endl;
}