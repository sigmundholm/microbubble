#ifndef MICROBUBBLE_CUTFEM_HEAT_EQUATION_H
#define MICROBUBBLE_CUTFEM_HEAT_EQUATION_H

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

#include "rhs.h"


using namespace dealii;
using namespace cutfem;


namespace examples::cut::HeatEquation {

    using NonMatching::LocationToLevelSet;


    template<int dim>
    class HeatEqn : public ScalarProblem<dim> {
    public:
        HeatEqn(const double nu,
                const double tau,
                const double radius,
                const double half_length,
                const unsigned int n_refines,
                const int element_order,
                const bool write_output,
                Function<dim> &rhs,
                Function<dim> &bdd_values,
                Function<dim> &analytical_soln,
                LevelSet<dim> &levelset_func,
                const bool stabilized = true,
                const bool crank_nicholson = false,
                const bool compute_error = true);

        static void
        write_header_to_file(std::ofstream &file);

        static void
        write_error_to_file(ErrorBase *error, std::ofstream &file);

    protected:
        void
        set_function_times(double time) override;

        void
        make_grid(Triangulation<dim> &tria) override;

        virtual void
        assemble_local_over_cell(const FEValues<dim> &fe_values,
                                 const std::vector<types::global_dof_index> &loc2glb) override;

        virtual void
        assemble_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glb) override;

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
        assemble_rhs(int time_step) override;

        void // TODO not a non-cn version of this method?
        assemble_rhs_local_over_cell_cn(const FEValues<dim> &fe_values,
                                        const std::vector<types::global_dof_index> &loc2glb,
                                        const int time_step) override;

        void
        assemble_rhs_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glob) override;

        void
        assemble_rhs_local_over_surface_cn(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glob,
                const int time_step) override;


        const double nu;

        const double radius;
        const double half_length;

    };

} // namespace examples::cut::HeatEquation

#endif //MICROBUBBLE_CUTFEM_HEAT_EQUATION_H
