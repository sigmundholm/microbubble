#ifndef MICROBUBBLE_FLOW_PROBLEM_H
#define MICROBUBBLE_FLOW_PROBLEM_H

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

#include "cutfem_problem.h"


using namespace dealii;
using namespace cutfem;

namespace utils::problems::flow {

    using NonMatching::LocationToLevelSet;
    using namespace utils::problems;

    struct ErrorFlow : ErrorBase {
        double l2_error_u = 0;
        double h1_error_u = 0;
        double h1_semi_u = 0;
        double l2_error_p = 0;
        double h1_error_p = 0;
        double h1_semi_p = 0;
        double l_inf_l2_error_u = 0;
        double l_inf_h1_error_u = 0;

        void output() override {
            std::cout << "  k = " << time_step << ", "
                      << "|| u - u_h ||_L2 = " << l2_error_u
                      << ", || u - u_h ||_H1 = " << h1_error_u
                      << ", || p - p_h ||_L2 = " << l2_error_p
                      << ", || p - p_h ||_H1 = " << h1_error_p
                      << std::endl;
        }
    };


    template<int dim>
    class FlowProblem : public CutFEMProblem<dim> {
    public:
        FlowProblem(const unsigned int n_refines,
                    const int element_order,
                    const bool write_output,
                    LevelSet<dim> &levelset_func,
                    TensorFunction<1, dim> &analytic_v,
                    Function<dim> &analytic_p,
                    const bool stabilized = true);

        static void
        write_header_to_file(std::ofstream &file);

        static void
        write_error_to_file(ErrorBase *error, std::ofstream &file);

    protected:
        virtual void
        interpolate_solution(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                             int time_step) override;

        virtual void
        setup_fe_collection() override;


        virtual void
        assemble_system() override;

        virtual void
        assemble_rhs_and_bdf_terms_local_over_cell(
                const FEValues<dim> &fe_v,
                const std::vector<types::global_dof_index> &loc2glb) override;

        virtual void
        assemble_rhs_and_bdf_terms_local_over_cell_moving_domain(
                const FEValues<dim> &fe_v,
                const std::vector<types::global_dof_index> &loc2glb) override;


        ErrorBase *
        compute_error(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                      Vector<double> &solution) override;

        ErrorBase *
        compute_time_error(std::vector<ErrorBase *> &errors) override;

        void
        integrate_cell(const FEValues<dim> &fe_v,
                       Vector<double> &solution,
                       double &l2_error_integral_u,
                       double &h1_error_integral_u,
                       double &l2_error_integral_p,
                       double &h1_error_integral_p,
                       const double &mean_numerical_pressure,
                       const double &mean_exact_pressure) const;


        virtual void
        write_time_header_to_file(std::ofstream &file) override;

        virtual void
        write_time_error_to_file(ErrorBase *error,
                                 std::ofstream &file) override;


        virtual void
        output_results(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                       Vector<double> &solution,
                       std::string &suffix,
                       bool minimal_output = false) const override;

        FESystem<dim> mixed_fe;

        TensorFunction<1, dim> *rhs_function;
        TensorFunction<1, dim> *boundary_values;
        TensorFunction<1, dim> *analytical_velocity;
        Function<dim> *analytical_pressure;
};

} // namespace utils::problems::flow


#endif // MICROBUBBLE_FLOW_PROBLEM_H
