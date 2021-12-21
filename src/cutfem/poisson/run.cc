#include "poisson.h"
#include "rhs.h"

#include "cutfem/geometry/SignedDistanceSphere.h"

using namespace cutfem;

using namespace utils::problems::scalar;

int main(int argc, char *argv[]) {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const int dim = 2;
    double radius = 1;
    double half_length = 1;
    int n_refines = 5;
    int degree = 1;
    bool write_output = true;

    RightHandSide<dim> rhs;
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

    Poisson<dim> poisson(radius, half_length, n_refines, degree, write_output,
                         rhs, bdd, soln, domain);

    ErrorBase *err = poisson.run_step();
    auto *error = dynamic_cast<ErrorScalar*>(err);
    error->output();
}