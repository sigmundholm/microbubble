#include <deal.II/base/point.h>

#include <deal.II/numerics/vector_tools.h>

#include "ns_benchmark.h"

using namespace cutfem;

namespace examples::cut::NavierStokes::benchmarks {

    template<int dim>
    BenchmarkNS<dim>::
    BenchmarkNS(const double nu, const double tau, const double radius,
                const double half_length, const unsigned int n_refines,
                const int element_order, const bool write_output,
                TensorFunction<1, dim> &rhs,
                ParabolicFlow<dim> &bdd_values,
                TensorFunction<1, dim> &analytic_vel,
                Function<dim> &analytic_pressure,
                Sphere<dim> &levelset_func,
                const bool semi_implicit, const int do_nothing_id,
                const bool stabilized, const bool stationary,
                const bool compute_error)
            : NavierStokes::NavierStokesEqn<dim>(
            nu, tau, radius, half_length, n_refines, element_order,
            write_output, rhs, bdd_values, analytic_vel, analytic_pressure,
            levelset_func, semi_implicit, do_nothing_id, stabilized,
            stationary, compute_error) {}


    template<int dim>
    void BenchmarkNS<dim>::
    post_processing() {
        std::cout << "Post-processing:" << std::endl;

        // Compute the surface forces. Which are the drag and lift forces
        // respectively. Then these are used for computing the drag and lift
        // coefficients.
        Tensor<1, dim> surface_forces = this->compute_surface_forces();

        auto *bdd_values = dynamic_cast<ParabolicFlow<dim> *>(this->boundary_values);
        double max_velocity = bdd_values->get_current_max_speed();
        double average_speed = 2.0 / 3 * max_velocity;

        auto *sphere = dynamic_cast<Sphere<dim> *>(this->levelset_function);
        double length = 2 * sphere->get_radius();

        // Compute the drag and lift coefficients.
        double factor = 2 / (pow(average_speed, 2) * length);
        double drag_coefficient = factor * surface_forces[0];
        double lift_coefficient = factor * surface_forces[1];

        std::cout << " * Drag: C_D = " << drag_coefficient << std::endl;
        std::cout << " * Lift: C_L = " << lift_coefficient << std::endl;

        double pressure_diff = compute_pressure_difference();
        std::cout << " * Δp = " << pressure_diff << std::endl;
    }

    template<int dim>
    double BenchmarkNS<dim>::
    compute_pressure_difference() {
        Point<dim> a1(0.15 - this->half_length, 0.2 - this->radius);
        Point<dim> a2(0.25 - this->half_length, 0.2 - this->radius);

        // TODO Use a method that takes a mapping to get higher than only
        //  Q_1 elements: see VectorTools.
        Vector<double> value_a1(dim + 1);
        VectorTools::point_value(*this->dof_handlers.front(),
                                 this->solutions.front(), a1, value_a1);

        Vector<double> value_a2(dim + 1);
        VectorTools::point_value(*this->dof_handlers.front(),
                                 this->solutions.front(), a2, value_a2);
        return value_a1[dim] - value_a2[dim];
    }

    template
    class BenchmarkNS<2>;

}
