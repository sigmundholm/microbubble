#ifndef MICROBUBBLE_STOKES_GEN_H
#define MICROBUBBLE_STOKES_GEN_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
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

#include "../utils/flow_problem.h"
#include "rhs.h"


using namespace dealii;
using namespace cutfem;

namespace examples::cut::StokesEquation2 {

    using NonMatching::LocationToLevelSet;
    using namespace utils::problems::flow;

    template<int dim>
    class StokesEqn2 : public FlowProblem<dim> {
    public:
        StokesEqn2(const double radius,
                   const double half_length,
                   const unsigned int n_refines,
                   const double delta,
                   const double nu,
                   const double tau,
                   const int element_order,
                   const bool write_output,
                   TensorFunction<1, dim> &rhs,
                   TensorFunction<1, dim> &bdd_values,
                   TensorFunction<1, dim> &analytic_vel,
                   Function<dim> &analytic_pressure,
                   LevelSet<dim> &levelset_func);

        static void
        write_header_to_file(std::ofstream &file);

        static void
        write_error_to_file(ErrorBase *error, std::ofstream &file);

    protected:
        void
        set_function_times(double time) override;


        void
        make_grid(Triangulation<dim> &tria) override;

        void
        assemble_system() override;

        void
        assemble_matrix() override;

        void
        assemble_matrix_local_over_cell(const FEValues<dim> &fe_values,
                                 const std::vector<types::global_dof_index> &loc2glb) override;

        void
        assemble_matrix_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glb) override;

        void
        assemble_rhs(int time_step, bool moving_domain) override;

        void
        assemble_rhs_local_over_surface(
                const FEValuesBase <dim> &fe_values,
                const std::vector <types::global_dof_index> &loc2glb) override;

        const double radius;
        const double half_length;

        const double delta;
        const double nu;

        unsigned int do_nothing_id = 10;

    };

} // namespace


#endif // MICROBUBBLE_STOKES_GEN_H
