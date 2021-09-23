#ifndef MICROBUBBLE_CUTFEM_PROBLEM_H
#define MICROBUBBLE_CUTFEM_PROBLEM_H

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


namespace utils::problems {

    using namespace dealii;
    using namespace cutfem;

    using NonMatching::LocationToLevelSet;


    struct ErrorBase {
        double h = 0;
        double tau = 0;
        double time_step = 0;
        double cond_num = 0;

        virtual void output() {
            std::cout << "h = " << h << std::endl;
        }

        // virtual void file_header(std::ofstream &file, bool time_dependent);

        // virtual void file_output(std::ofstream &file, bool time_dependent);
    };


    template<int dim>
    class CutFEMProblem {
    public:
        CutFEMProblem(const unsigned int n_refines,
                      const int element_order,
                      const bool write_output,
                      Function<dim> &levelset_func,
                      const bool stabilized = true);


        ErrorBase
        run_step();

        Vector<double>
        get_solution();

        /**
         * Run a time loop with a BDF-method.
         *
         * The initial step u_0 is interpolated from the object boundary_values
         * passes as the argument bdd_values in the constructor. If u1 is a
         * vector of length exactly 1, then u1 is also interpolated. Else, the
         * argument u1 is ´used as the start up step u1 for BDF-2.
         *
         * @param u1: is the u1 start up step.
         * @param steps: the number of steps to run.
         * @return an Error object.
         */
        ErrorBase
        run_time(unsigned int bdf_type, unsigned int steps,
                 std::vector<Vector<double>> &supplied_solutions);

        ErrorBase
        run_time(unsigned int bdf_type, unsigned int steps);

        static void
        write_header_to_file(std::ofstream &file);

        static void
        write_error_to_file(ErrorBase &error, std::ofstream &file);

    protected:
        void
        set_bdf_coefficients(unsigned int bdf_type);

        void
        interpolate_first_steps(unsigned int bdf_type,
                                std::vector<ErrorBase> &errors);

        void
        set_supplied_solutions(unsigned int bdf_type,
                               std::vector<Vector<double>> &supplied_solutions,
                               std::vector<ErrorBase> &errors);

        virtual void
        set_function_times(double time);

        virtual void
        interpolate_solution(int time_step);


        virtual void
        make_grid(Triangulation<dim> &tria) = 0;

        virtual void
        setup_quadrature();

        virtual void
        setup_level_set();

        virtual void
        distribute_dofs() = 0;

        virtual void
        initialize_matrices();


        virtual void
        assemble_system() = 0;

        virtual void
        assemble_local_over_cell(const FEValues<dim> &fe_values,
                                 const std::vector<types::global_dof_index> &loc2glb) = 0;

        virtual void
        assemble_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glb) = 0;


        virtual void
        assemble_matrix() = 0;

        virtual void
        assemble_matrix_local_over_cell(const FEValues<dim> &fe_values,
                                        const std::vector<types::global_dof_index> &loc2glb) = 0;

        virtual void
        assemble_matrix_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glb) = 0;

        virtual void
        assemble_rhs(int time_step) = 0;

        virtual void
        assemble_rhs_local_over_cell(const FEValues<dim> &fe_values,
                                     const std::vector<types::global_dof_index> &loc2glb) = 0;

        virtual void
        assemble_rhs_local_over_cell_cn(const FEValues<dim> &fe_values,
                                        const std::vector<types::global_dof_index> &loc2glb,
                                        const int time_step) = 0;

        virtual void
        assemble_rhs_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glob) = 0;

        virtual void
        assemble_rhs_local_over_surface_cn(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glob,
                const int time_step) = 0;


        virtual void
        solve();


        virtual ErrorBase
        compute_error() = 0;

        virtual ErrorBase
        compute_time_error(std::vector<ErrorBase> errors) = 0;

        virtual void
        integrate_cell(const FEValues<dim> &fe_v,
                       double &l2_error_integral,
                       double &h1_error_integral) const = 0;

        double
        compute_condition_number();


        virtual void
        write_time_header_to_file(std::ofstream &file) = 0;

        virtual void
        write_time_error_to_file(ErrorBase &error, std::ofstream &file) = 0;


        virtual void
        output_results(std::string &suffix,
                       bool minimal_output = false) const = 0;

        virtual void
        output_results(int time_step,
                       bool minimal_output = false) const;

        virtual void
        output_results(bool minimal_output = false) const;

        const unsigned int n_refines;
        const unsigned int element_order;
        bool write_output;

        double h;  // cell side length
        double tau;

        Triangulation<dim> triangulation;

        hp::FECollection<dim> fe_collection;
        hp::MappingCollection<dim> mapping_collection;
        hp::QCollection<dim> q_collection;
        hp::QCollection<1> q_collection1D;

        // Object managing degrees of freedom for the level set function.
        FE_Q<dim> fe_levelset;
        DoFHandler<dim> levelset_dof_handler;
        Vector<double> levelset;

        Function<dim> *rhs_function;
        Function<dim> *boundary_values;
        Function<dim> *levelset_function;

        // Object managing degrees of freedom for the cutfem method.
        hp::DoFHandler<dim> dof_handler;

        NonMatching::CutMeshClassifier<dim> cut_mesh_classifier;

        SparsityPattern sparsity_pattern;

        SparseMatrix<double> stiffness_matrix;
        Vector<double> rhs;
        Vector<double> solution;

        AffineConstraints<double> constraints;

        // Vector of previous solutions, used in the time discretization method.
        std::vector<Vector<double>> solutions; // (u0, u1) when BDF-2 is used.

        // Constants used for the time discretization, defined as:
        //   u_t = (au^(n+1) + bu^n + cu^(n-1))/τ, where u = u^(n+1)
        // For BDF-1: (b, a), and (c, b, a) for BDF-2.
        std::vector<double> bdf_coeffs;
        bool crank_nicholson;

        const bool stabilized;

    };
}

#endif //MICROBUBBLE_CUTFEM_PROBLEM_H
