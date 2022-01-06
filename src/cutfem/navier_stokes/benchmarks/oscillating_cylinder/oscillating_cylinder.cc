#include <cmath>

#include "rhs.h"

#include "../ns_benchmark.h"

/**
 * This program performes the benchmark test for Navier-Stokes on a moving
 * domain, as proposed by Balmus-Massing-Hoffman (2020). 
 */
int main() {
    using namespace examples::cut::NavierStokes;
    using namespace utils::problems::flow;

    const int dim = 2;

    const int n_refines = 5;
    const int element_order = 1;
    const bool write_vtk = true;

    const bool semi_implicit = true;

    const double radius = 0.205;
    const double half_length = 1.1;
    const double sphere_radius = 0.05;

    const double end_time = 2.0;
    const double tau = 0.05;
    const unsigned int time_steps = end_time / tau;

    const double nu = 0.001;

    const double amplitude = 0.2;
    const double omega = 2 * M_PI;
    
    benchmarks::oscillating::OscillatingSphere<dim> domain(sphere_radius, 0.0, -0.005, amplitude, omega);

    benchmarks::oscillating::BoundaryValues<dim> boundary(
        half_length, radius, sphere_radius, domain);

    ZeroTensorFunction<1, dim> zero_tensor;
    Functions::ZeroFunction<dim> zero_scalar;

    benchmarks::BenchmarkNS<dim> ns(
            nu, tau, radius, half_length, n_refines, element_order, write_vtk,
            zero_tensor, boundary, zero_tensor, zero_scalar,
            domain, "benchmark-2D-3.csv", semi_implicit, {0, 1}, false, true);

    // BDF-1
    ns.run_moving_domain(1, 1);

    Vector<double> u1 = ns.get_solution();
    std::vector<Vector<double>> initial = {u1};
    std::shared_ptr<hp::DoFHandler<dim>> u1_dof = ns.get_dof_handler();
    std::vector<std::shared_ptr<hp::DoFHandler<dim>>> initial_dofs = {u1_dof};

    // BDF-2
    ns.run_moving_domain(2, time_steps, initial, initial_dofs);
}

