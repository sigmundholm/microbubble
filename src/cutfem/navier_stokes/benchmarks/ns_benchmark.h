#ifndef MICROBUBBLE_NS_BENCHMARK_H
#define MICROBUBBLE_NS_BENCHMARK_H

#include "../navier_stokes.h"
#include "../rhs.h"

using namespace dealii;
using namespace cutfem;


namespace examples::cut::NavierStokes::benchmarks {

    using namespace examples::cut;

    /**
     * Class for computing the benchmark tests 2D-1 and 2D-3 at
     *   http://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow.html
     *
     * This class implements the method post_processing to compute the pressure
     * difference at the desired points for the tests, and also the drag and
     * lift coefficients.
     *
     * This test is performed for parabolic flow around a cylinder in a channel
     * (this implementation implements the 2D case).
     *
     * @tparam dim
     */
    template<int dim>
    class BenchmarkNS : public NavierStokes::NavierStokesEqn<dim> {
    public:
        BenchmarkNS(double nu, double tau, double radius, double half_length,
                    unsigned int n_refines, int element_order,
                    bool write_output,
                    TensorFunction<1, dim> &rhs,
                    ParabolicFlow<dim> &bdd_values,
                    TensorFunction<1, dim> &analytic_vel,
                    Function<dim> &analytic_pressure,
                    Sphere<dim> &levelset_func,
                    std::string filename,
                    bool semi_implicit, 
                    std::vector<int> do_nothing_ids,
                    bool stationary = false, bool compute_error = true);

    protected:
        void make_grid(Triangulation<dim> &tria);

        /**
         * Compute the pressure difference between two points in front of and
         * behind the sphere. Also compute the drag and lift coefficients of
         * sphere.
         */
        void post_processing(unsigned int time_step);

        /**
         * Compute the pressure difference between the points a_1 and a_2, used in
         * the benchmarks 2D-1 and 2D-3.
         * @tparam dim
         * @return
         */
        double compute_pressure_difference();

    private:
        // Write the computed data to file as csv.
        std::ofstream file;

    };
}

#endif //MICROBUBBLE_NS_BENCHMARK_H
