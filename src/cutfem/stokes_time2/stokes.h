#ifndef MICROBUBBLE_STOKES_BDF2_H
#define MICROBUBBLE_STOKES_BDF2_H

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
#include "rhs.h"


using namespace dealii;
using namespace cutfem;


namespace examples::cut::StokesEquation {

    using NonMatching::LocationToLevelSet;
    using namespace utils::problems::flow;


    template<int dim>
    class StokesEqn : public FlowProblem<dim> {
    public:
        StokesEqn(double nu,
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
                  int do_nothing_id = 10,
                  bool stabilized = true,
                  bool stationary = false,
                  bool compute_error = true);

        static void
        write_header_to_file(std::ofstream &file);

        static void
        write_error_to_file(ErrorBase *error, std::ofstream &file);

        /**
         * Compute the stress over the surface of the submerged body (over the
         * level set function boundary).
         *
         * The computation methods (Stress::<field>) can be combined by using
         * the bitwise or | operator.
         *
         * @param method as a field of the Stress enum defined in FlowProblem.h.
         *   Can also use the bitwise or operator | to apply multiple Stress
         *   settings for the computation. The different choices are:
         *    - Stress::Regular: compute the stress by using that
         *          σ = ν𝛁u - pI
         *    - Stress::Symmetric: compute the stress by using the symmetric
         *        gradient, s.t.
         *          σ = ν(𝛁u + 𝛁u^T)/2 - pI
         *    - Stress::NitscheFlux: compute the stress by adjusting for the
         *        Nitche terms, by integraing
         *          ν𝛁u - pI + 𝛾/h(u - g)
         *        over the surface. When combined with Stress::Symmetric, the
         *        symmetric gradient is used.
         *    - Stress::Exact: compute the stress using the exact solution in
         *        the quadrature points.
         *    - Stress::Test: this flag should be used when performing a
         *        convergence test for the different computation methods. This
         *        causes the mean pressure over the boundary to be subtracted
         *        from the pressure value, since the computed pressure is not
         *        unique when using Dirichlet on the whole boundary.
         * @return the stress (drag and lift)
         */
        Tensor<1, dim>
        compute_surface_forces(unsigned int method = Stress::Regular);

    protected:
        void
        set_function_times(double time) override;

        void
        make_grid(Triangulation<dim> &tria) override;

        void
        pre_matrix_assembly() override;

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

        void
        assemble_rhs_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glb) override;

        void
        integrate_surface_forces(const FEValuesBase<dim> &fe_v,
                                 Vector<double> solution,
                                 unsigned int method,
                                 Tensor<1, dim> &viscous_forces,
                                 Tensor<1, dim> &pressure_forces);

        const double nu;

        const double radius;
        const double half_length;

        unsigned int do_nothing_id;

    };

} // namespace examples::cut::StokesEquation


#endif // MICROBUBBLE_STOKES_BDF2_H
