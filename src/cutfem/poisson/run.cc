#include "poisson.h"
#include "rhs.h"


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

    // FlowerDomain<dim> domain;
    Sphere<dim> domain(sphere_radius, 0, 0);

    Poisson<dim> poisson(radius, half_length, n_refines, degree, write_output,
                         rhs, bdd, soln, domain);

    ErrorBase *err = poisson.run_step();
    auto *error = dynamic_cast<ErrorScalar*>(err);
    error->output();
}