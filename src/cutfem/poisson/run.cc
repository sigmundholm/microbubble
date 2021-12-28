#include "poisson.h"
#include "rhs.h"


using namespace cutfem;

using namespace utils::problems::scalar;

int main() {
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

    FlowerDomain<dim> domain;

    Poisson<dim> poisson(radius, half_length, n_refines, degree, write_output,
                         rhs, bdd, soln, domain);

    ErrorBase *err = poisson.run_step();
    auto *error = dynamic_cast<ErrorScalar*>(err);
    error->output();
}