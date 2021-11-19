#ifndef MICROBUBBLE_NAVIER_STOKES_H
#define MICROBUBBLE_NAVIER_STOKES_H

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

#include "../utils/flow_problem.h"
#include "../stokes_time2/stokes.h"
#include "rhs.h"


using namespace dealii;
using namespace cutfem;


namespace examples::cut::NavierStokes {

    using namespace examples::cut;


    template<int dim>
    class NavierStokesEqn : public StokesEquation::StokesEqn<dim> {
    public:
        NavierStokesEqn(double nu,
                        double tau,
                        double radius,
                        double half_length,
                        unsigned int n_refines,
                        int element_order,
                        bool write_output,
                        TensorFunction<1, dim> &rhs,
                        TensorFunction<1, dim> &bdd_values,
                        TensorFunction<1, dim> &analytic_vel,
                        Function<dim> &analytic_pressure,
                        LevelSet<dim> &levelset_func,
                        bool semi_implicit,
                        int do_nothing_id = 10,
                        bool stabilized = true,
                        bool stationary = false,
                        bool compute_error = true);

    protected:
        void
        set_function_times(double time) override;

        void
        make_grid(Triangulation<dim> &tria) override;

        void
        pre_matrix_assembly() override;

        void
        assemble_timedep_matrix() override;

        void
        assemble_matrix_local_over_cell(
                const FEValues<dim> &fe_v,
                const std::vector<types::global_dof_index> &loc2glb) override;

        void
        assemble_convection_over_cell(
                const FEValues<dim> &fe_v,
                const std::vector<types::global_dof_index> &loc2glb);

        void
        assemble_convection_over_cell_moving_domain(
                const FEValues<dim> &fe_v,
                const std::vector<types::global_dof_index> &loc2glb);

        void
        assemble_rhs(int time_step) override;

        void
        assemble_rhs_and_bdf_terms_local_over_cell(
                const FEValues<dim> &fe_v,
                const std::vector<types::global_dof_index> &loc2glb) override;

        void
        assemble_rhs_and_bdf_terms_local_over_cell_moving_domain(
                const FEValues<dim> &fe_v,
                const std::vector<types::global_dof_index> &loc2glb) override;

        Tensor<1, dim>
        compute_surface_forces();

        void
        integrate_surface_forces(const FEValuesBase<dim> &fe_v,
                                 Vector<double> solution,
                                 Tensor<1, dim> &viscous_forces,
                                 Tensor<1, dim> &pressure_forces);


        // If true, a semi-implicit discretisation is used for the convection
        // term. Else, it an explicit discretisation is used.
        const bool semi_implicit;

    };

} // namespace examples::cut::NavierStokes


#endif //MICROBUBBLE_NAVIER_STOKES_H
