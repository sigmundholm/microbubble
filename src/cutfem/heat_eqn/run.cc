#include "heat_eqn.h"
#include "rhs.h"

using namespace cutfem;

using namespace examples::cut::HeatEquation;


int main() {
    const int dim = 2;
    double radius = 1;
    double half_length = 2 * radius;
    int n_refines = 6;
    int degree = 1;
    bool write_output = true;

    const double nu = 2;
    const double end_time = 4;
    const double tau = 0.02;

    const double time_steps = end_time / tau;

    RightHandSide<dim> rhs(nu, tau);
    BoundaryValues<dim> bdd;
    AnalyticalSolution<dim> soln;

    double sphere_radius = radius * 0.75;
    double sphere_x_coord = 0;
    Point<dim> sphere_center;
    if (dim == 2) {
        sphere_center = Point<dim>(0, 0);
    } else if (dim == 3) {
        sphere_center = Point<dim>(0, 0, 0);
    }
    
    FlowerDomain<dim> domain(0.6, 3.5);
    // MovingDomain<dim> domain(sphere_radius, half_length, radius);

    HeatEqn<dim> heat(nu, tau, radius, half_length, n_refines, degree,
                      write_output, rhs, bdd, soln, domain);

    // BDF-1
    ErrorBase *err = heat.run_moving_domain(1, 1, 2);

    // BDF-2
    Vector<double> u1 = heat.get_solution();
    std::shared_ptr<hp::DoFHandler<dim>> u1_dof_h = heat.get_dof_handler();
    std::vector<Vector<double>> initial = {u1};
    std::vector<std::shared_ptr<hp::DoFHandler<dim>>> initial_dof_h = {
            u1_dof_h};
    ErrorBase *err2 = heat.run_moving_domain(2, time_steps,
                                             initial, initial_dof_h);


    auto *error = dynamic_cast<ErrorScalar *>(err);
    std::cout << "|| u - u_h ||_L2 = " << error->l2_error << std::endl;
    std::cout << "|| u - u_h ||_H1 = " << error->h1_error << std::endl;
    std::cout << "| u - u_h |_H1 = " << error->h1_semi << std::endl;
}

