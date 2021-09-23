#ifndef MICROBUBBLE_SCALAR_PROBLEM_H
#define MICROBUBBLE_SCALAR_PROBLEM_H

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

#include "cutfem/geometry/SignedDistanceSphere.h"
#include "cutfem/nla/sparsity_pattern.h"
#include "cutfem/stabilization/jump_stabilization.h"

#include "cutfem_problem.h"


using namespace dealii;
using namespace cutfem;


namespace utils::problems::scalar {

    using NonMatching::LocationToLevelSet;

    using namespace utils::problems;


    struct ErrorScalar : ErrorBase {
        double h = 0;
        double tau = 0;
        double time_step = 0;
        double l2_error = 0;
        double h1_error = 0;
        double h1_semi = 0;
        double l_inf_l2_error = 0;
        double l_inf_h1_error = 0;
        double cond_num = 0;

        void output() override {
            std::cout << "  k = " << time_step << ", "
                      << "|| u - u_h ||_L2 = " << l2_error
                      << ", || u - u_h ||_H1 = " << h1_error
                      << std::endl;
        }

        /*
        // TODO these or delete
        void file_header(std::ofstream &file, bool time_dependent) {
        }

        void file_output(std::ofstream &file, bool time_dependent) {
            if (time_dependent) {

            }
        }
         */

    };


    template<int dim>
    class ScalarProblem : public CutFEMProblem<dim> {
    public:
        ScalarProblem(const unsigned int n_refines,
                      const int element_order,
                      const bool write_output,
                      Function<dim> &levelset_func,
                      Function<dim> &analytical_soln,
                      const bool stabilized = true);

        static void
        write_header_to_file(std::ofstream &file);

        static void
        write_error_to_file(ErrorBase &error, std::ofstream &file);

    protected:
        virtual void
        interpolate_solution(int time_step) override;

        void
        distribute_dofs() override;

        virtual void
        assemble_system() override;


        ErrorBase
        compute_error() override;

        ErrorBase
        compute_time_error(std::vector<ErrorBase> errors) override;

        void
        integrate_cell(const FEValues<dim> &fe_v,
                       double &l2_error_integral,
                       double &h1_error_integral) const override;


        virtual void
        write_time_header_to_file(std::ofstream &file) override;

        virtual void
        write_time_error_to_file(ErrorBase &error, std::ofstream &file) override;


        virtual void
        output_results(std::string &suffix,
                       bool minimal_output = false) const override;

        FE_Q<dim> fe;

        Function<dim> *analytical_solution;

    };

} // namespace utils::problems::scalar


#endif //MICROBUBBLE_SCALAR_PROBLEM_H
