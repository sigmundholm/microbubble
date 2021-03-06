#include "poisson.h"
#include "rhs.h"

int main() {
    const int dim = 2;
    double radius = 1;
    double half_length = 1;
    int n_refines = 5;
    int degree = 1;
    bool write_output = true;
    double sphere_rad = radius * 0.9;
    double sphere_x_coord = 0;

    RightHandSide<dim> rhs;
    BoundaryValues<dim> bdd;
    AnalyticalSolution<dim> soln;

    Poisson<dim> poisson(radius, half_length, n_refines, degree, write_output,
                         rhs, bdd, soln, sphere_rad, sphere_x_coord);

    poisson.run();
}