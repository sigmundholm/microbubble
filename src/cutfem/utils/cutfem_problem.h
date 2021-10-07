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

#include <deque>
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
    };


    template<int dim>
    class CutFEMProblem {
    public:
        CutFEMProblem(const unsigned int n_refines,
                      const int element_order,
                      const bool write_output,
                      Function<dim> &levelset_func,
                      const bool stabilized = true);


        ErrorBase *
        run_step();

        Vector<double>
        get_solution();

        hp::DoFHandler<dim> &
        get_dof_handler();

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
        ErrorBase *
        run_time(unsigned int bdf_type, unsigned int steps,
                 std::vector<Vector<double>> &supplied_solutions);

        ErrorBase *
        run_time(unsigned int bdf_type, unsigned int steps);

        ErrorBase *
        run_moving_domain(unsigned int bdf_type, unsigned int steps,
                          std::vector<Vector<double>> &supplied_solutions,
                          std::vector<std::reference_wrapper<hp::DoFHandler<dim>>> &supplied_dof_handlers);

        ErrorBase *
        run_moving_domain(unsigned int bdf_type, unsigned int steps);

        static void
        write_header_to_file(std::ofstream &file);

        static void
        write_error_to_file(ErrorBase *error, std::ofstream &file);

    protected:
        void
        set_bdf_coefficients(unsigned int bdf_type);

        void
        interpolate_first_steps(unsigned int bdf_type,
                                std::vector<ErrorBase *> &errors,
                                bool moving_domain = false);

        void
        set_supplied_solutions(unsigned int bdf_type,
                               std::vector<Vector<double>> &supplied_solutions,
                               std::vector<std::reference_wrapper<hp::DoFHandler<dim>>> &supplied_dof_handlers,
                               std::vector<ErrorBase *> &errors,
                               bool moving_domain = false);

        virtual void
        set_function_times(double time);

        virtual void
        interpolate_solution(hp::DoFHandler<dim> &dof_handler,
                             int time_step, bool moving_domain = false);


        virtual void
        make_grid(Triangulation<dim> &tria) = 0;

        virtual void
        setup_quadrature();

        virtual void
        setup_level_set();

        virtual void
        setup_fe_collection() = 0;

        virtual void
        distribute_dofs(hp::DoFHandler<dim> &dof_handler,
                        double size_of_bound = 0);

        virtual void
        initialize_matrices();


        // Methods related to assembling the stiffness matrix and rhs vector.
        // -------------------------------------------------------------------

        /**
         * Assemble the stiffness matrix and rhs vector. This method is used
         * for stationary problems.
         *
         * This method should in turn call the methods assemble_local_over_cell
         * and assemble_local_over_surface.
         */
        virtual void
        assemble_system();

        virtual void
        assemble_local_over_cell(const FEValues<dim> &fe_values,
                                 const std::vector<types::global_dof_index> &loc2glb);

        virtual void
        assemble_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glb);

        // TODO create a method assemble_rhs() for stationary problems.

        virtual void
        assemble_matrix();

        virtual void
        assemble_matrix_local_over_cell(const FEValues<dim> &fe_values,
                                        const std::vector<types::global_dof_index> &loc2glb);

        virtual void
        assemble_matrix_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glb);

        virtual void
        assemble_rhs(int time_step, bool moving_domain);

        virtual void
        assemble_rhs_local_over_cell(const FEValues<dim> &fe_values,
                                     const std::vector<types::global_dof_index> &loc2glb);

        virtual void
        assemble_rhs_and_bdf_terms_local_over_cell(
                const FEValues<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glb) = 0;

        virtual void
        assemble_rhs_and_bdf_terms_local_over_cell_moving_domain(
                const FEValues<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glb) = 0;

        virtual void
        assemble_rhs_local_over_cell_cn(const FEValues<dim> &fe_values,
                                        const std::vector<types::global_dof_index> &loc2glb,
                                        const int time_step);

        virtual void
        assemble_rhs_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glob);

        virtual void
        assemble_rhs_local_over_surface_cn(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glob,
                const int time_step);


        virtual void
        solve();

        virtual ErrorBase *
        compute_error(hp::DoFHandler<dim> &dof_handler,
                      Vector<double> &solution) = 0;

        virtual ErrorBase *
        compute_time_error(std::vector<ErrorBase *> &errors) = 0;

        double
        compute_condition_number();


        virtual void
        write_time_header_to_file(std::ofstream &file) = 0;

        virtual void
        write_time_error_to_file(ErrorBase *error, std::ofstream &file) = 0;


        virtual void
        output_results(hp::DoFHandler<dim> &dof_handler,
                       Vector<double> &solution,
                       std::string &suffix,
                       bool minimal_output = false) const = 0;

        virtual void
        output_results(hp::DoFHandler<dim> &dof_handler,
                       Vector<double> &solution,
                       int time_step,
                       bool minimal_output = false) const;

        virtual void
        output_results(hp::DoFHandler<dim> &dof_handler,
                       Vector<double> &solution,
                       bool minimal_output = false) const;

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

        Function<dim> *levelset_function;

        // Object managing degrees of freedom for the cutfem method.
        std::deque<hp::DoFHandler<dim>> dof_handlers;

        NonMatching::CutMeshClassifier<dim> cut_mesh_classifier;

        SparsityPattern sparsity_pattern;

        SparseMatrix<double> stiffness_matrix;
        Vector<double> rhs;
        // Vector<double> solution;

        AffineConstraints<double> constraints;

        // Queue of current and  previous solutions, used in the time
        // discretization method. When a new time step is solved, a new empty
        // solution vector is pushed to the front.
        //  - In the first iteration of BDF-2 it containts (u, u1, u0).
        std::deque<Vector<double>> solutions;

        // Constants used for the time discretization, defined as:
        //   u_t = (au^(n+1) + bu^n + cu^(n-1))/τ, where u = u^(n+1)
        // For BDF-1: (a, b), and (a, b, c) for BDF-2.
        std::vector<double> bdf_coeffs;
        bool crank_nicholson;

        const bool stabilized;

    };
}

#endif //MICROBUBBLE_CUTFEM_PROBLEM_H
