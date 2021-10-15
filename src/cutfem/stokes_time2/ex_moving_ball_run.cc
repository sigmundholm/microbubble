
#include "stokes.h"
#include "ex_moving_ball.h"

int main() {

    using namespace examples::cut::StokesEquation;

    const unsigned int n_refines = 4;
    const unsigned int elementOrder = 1;

    printf("numRefines=%d\n", n_refines);
    printf("elementOrder=%d\n", elementOrder);
    const bool write_vtk = true;

    double radius = 1;
    double half_length = 2 * radius;

    double nu = 1;
    double tau = 0.01;
    unsigned int n_steps = 100;

    double sphere_radius = radius * 0.75;

    const int dim = 2;

    ex1::ZeroTensorFunction<dim> zero_tensor;
    Functions::ZeroFunction<dim> zero_scalar;
    ex1::BoundaryValues<dim> boundary(sphere_radius, half_length, radius);
    ex1::MovingDomain<dim> domain(sphere_radius, half_length, radius);

    StokesEqn<dim> stokes(nu, tau, radius, half_length, n_refines,
                          elementOrder, write_vtk, zero_tensor, boundary,
                          zero_tensor, zero_scalar,
                          domain, 2);

    // BDF-1
    stokes.run_moving_domain(1, 1);

    // BDF-2
    Vector<double> u1 = stokes.get_solution();
    std::shared_ptr<hp::DoFHandler<dim>> dof = stokes.get_dof_handler();
    std::vector<Vector<double>> initial2 = {u1};
    std::vector<std::shared_ptr<hp::DoFHandler<dim>>> initial_dofh = {dof};

    stokes.run_moving_domain(2, n_steps, initial2, initial_dofh);
}


