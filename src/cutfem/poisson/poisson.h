#ifndef MICROBUBBLE_CUTFEM_POISSON_POISSON_H
#define MICROBUBBLE_CUTFEM_POISSON_POISSON_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_base.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <vector>

#include "cutfem/errors/error_calculator.h"

#include "rhs.h"
#include "../utils/scalar_problem.h"


using namespace dealii;
using namespace cutfem;


namespace cut::PoissonProblem {

    using NonMatching::LocationToLevelSet;
    using namespace utils::problems::scalar;

    template<int dim>
    class Poisson : public ScalarProblem<dim> {
    public:
        Poisson(double nu,
                double radius,
                double half_length,
                unsigned int n_refines,
                int element_order,
                bool write_output,
                Function<dim> &rhs,
                Function<dim> &bdd_values,
                Function<dim> &analytical_soln,
                Function<dim> &domain_func,
                bool stabilized = true);

        static void
        write_header_to_file(std::ofstream &file);

        static void
        write_error_to_file(ErrorBase *error, std::ofstream &file);

        double
        compute_surface_flux(unsigned int method);

    protected:
        void
        make_grid(Triangulation<dim> &tria) override;

        void
        pre_matrix_assembly() override;

        void
        assemble_local_over_cell(const FEValues<dim> &fe_values,
                                 const std::vector<types::global_dof_index> &loc2glb) override;

        void
        assemble_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glb) override;

        void
        integrate_surface_forces(const FEValuesBase<dim> &fe_v,
                                 Vector<double> solution,
                                 unsigned int method,
                                 double &flux_integral);

        const double nu;
        const double radius;
        const double half_length;
    };

} // namespace cut::PoissonProblem

#endif //MICROBUBBLE_CUTFEM_POISSON_POISSON_H
