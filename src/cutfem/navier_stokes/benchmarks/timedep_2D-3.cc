#include "ns_benchmark.h"
#include "../rhs_stat.h"

/**
 * This example is the 2D-1 benchmark example.
 * See: http://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html
 *
 * @return
 */
int main() {
    using namespace examples::cut::NavierStokes;
    using namespace utils::problems::flow;

    const unsigned int n_refines = 4;
    const unsigned int elementOrder = 1;

    printf("numRefines=%d\n", n_refines);
    printf("elementOrder=%d\n", elementOrder);
    const bool write_vtk = true;
    const int dim = 2;

    const double radius = 0.205;
    const double half_length = 1.1;

    const double end_time = 8.0;
    const double tau = 0.05;
    const unsigned int time_steps = end_time / tau;

    const double nu = 0.001;

    const bool semi_implicit = true;

    const double max_speed = 1.5;
    ParabolicFlow<dim> boundary(radius, half_length, max_speed, false);

    ZeroTensorFunction<1, dim> zero_tensor;
    Functions::ZeroFunction<dim> zero_scalar;

    const double sphere_radius = 0.05;
    Sphere<dim> domain(sphere_radius, -(half_length - 0.2), -0.005);

    benchmarks::BenchmarkNS<dim> ns(
            nu, tau, radius, half_length, n_refines, elementOrder, write_vtk,
            zero_tensor, boundary, zero_tensor, zero_scalar,
            domain, semi_implicit, 2, false, true);

    // BDF-1
    ns.run_time(1, 1);
    Vector<double> u1 = ns.get_solution();
    std::vector<Vector<double>> initial = {u1};

    // BDF-2
    ns.run_time(2, time_steps, initial);
}
