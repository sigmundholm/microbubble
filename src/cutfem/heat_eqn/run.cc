#include "heat_eqn.h"
#include "rhs.h"

#include "cutfem/geometry/SignedDistanceSphere.h"

using namespace cutfem;

using namespace examples::cut::HeatEquation;


int main() {
    const int dim = 2;
    double radius = 1;
    double half_length = 1;
    int n_refines = 5;
    int degree = 1;
    bool write_output = true;

    const double nu = 2;
    const double tau = 0.05;

    RightHandSide<dim> rhs(nu, tau);
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

    HeatEqn<dim> heat(nu, tau, radius, half_length, n_refines, degree, write_output,
                      rhs, bdd, soln, domain);

    Error error = heat.run(1, 40);
    std::cout << "|| u - u_h ||_L2 = " << error.l2_error << std::endl;
    std::cout << "|| u - u_h ||_H1 = " << error.h1_error << std::endl;
    std::cout << "| u - u_h |_H1 = " << error.h1_semi << std::endl;
}

