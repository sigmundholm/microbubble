#ifndef MICROBUBBLE_CUTFEM_PROBLEM_H
#define MICROBUBBLE_CUTFEM_PROBLEM_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

namespace LA 
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
!(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
using namespace dealii::LinearAlgebraPETSc;
# define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
using namespace dealii::LinearAlgebraTrilinos;
#else
# error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_base.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/fe_immersed_values.h>

#include <deal.II/numerics/data_out.h>

#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "stabilization/jump_stabilization.h"


namespace utils::problems {

    using namespace dealii;
    using namespace cutfem;

    using NonMatching::FEImmersedSurfaceValues;
    using NonMatching::LocationToLevelSet;


    struct ErrorBase {
        double h = 0;
        double tau = 0;
        double time_step = 0;
        double cond_num = 0;

        virtual void output() {
            std::cout << "h = " << h << std::endl;
        }

        /**
         * This method is used to get a representative error for the object.
         * This is used for the fixed point iteration.
         * @return
         */
        virtual double repr_error() {
            return 0;
        }
    };


    template<int dim>
    class LevelSet : public Function<dim> {
    public:
        virtual Tensor<1, dim>
        get_velocity();

        virtual double
        get_speed();
    };


    template<int dim>
    class CutFEMProblem {
    public:
        CutFEMProblem(unsigned int n_refines,
                      int element_order,
                      bool write_output,
                      LevelSet<dim> &levelset_func,
                      bool stabilized = true,
                      bool stationary = false,
                      bool compute_error = true);

        ErrorBase *
        run_step();

        ErrorBase *
        run_step_non_linear(double tol);

        LA::MPI::Vector
        get_solution();

        std::shared_ptr<hp::DoFHandler<dim>>
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
                 std::vector<LA::MPI::Vector> &supplied_solutions);

        ErrorBase *
        run_time(unsigned int bdf_type, unsigned int steps);

        ErrorBase *
        run_moving_domain(unsigned int bdf_type, unsigned int steps,
                          std::vector<LA::MPI::Vector> &supplied_solutions,
                          std::vector<std::shared_ptr<hp::DoFHandler<dim>>> &supplied_dof_handlers,
                          const double mesh_bound_multiplier = 1);

        ErrorBase *
        run_moving_domain(unsigned int bdf_type, unsigned int steps,
                          const double mesh_bound_multiplier = 1);

        void
        write_header_to_file(std::ofstream &file);

        void
        write_error_to_file(ErrorBase *error, std::ofstream &file);

    protected:
        void
        set_bdf_coefficients(unsigned int bdf_type);

        void
        set_extrapolation_coefficients(unsigned int bdf_type);

        void
        interpolate_first_steps(unsigned int bdf_type,
                                std::vector<ErrorBase *> &errors,
                                double mesh_bound_multiplier = 1);

        void
        set_supplied_solutions(unsigned int bdf_type,
                               std::vector<LA::MPI::Vector> &supplied_solutions,
                               std::vector<std::shared_ptr<hp::DoFHandler<dim>>> &supplied_dof_handlers,
                               std::vector<ErrorBase *> &errors);

        virtual void
        set_function_times(double time);

        virtual void
        interpolate_solution(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                             int time_step);


        virtual void
        make_grid(Triangulation<dim> &tria) = 0;

        void
        set_grid_size();

        virtual void
        setup_quadrature();

        virtual void
        setup_level_set();

        virtual void
        setup_fe_collection() = 0;

        virtual void
        distribute_dofs(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                        double size_of_bound = 0);

        void
        initialize_stiffness_matrix();

        void
        initialize_timedep_matrix();

        void
        make_sparsity_pattern_for_stabilized(
            DynamicSparsityPattern &dsp,
            const hp::DoFHandler<dim> &dof_handler);

        virtual void
        pre_matrix_assembly();


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
        assemble_timedep_matrix();

        virtual void
        assemble_matrix_local_over_cell(const FEValues<dim> &fe_values,
                                        const std::vector<types::global_dof_index> &loc2glb);

        virtual void
        assemble_matrix_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glb);

        virtual void
        assemble_rhs(int time_step);

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
        compute_error(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                      LA::MPI::Vector &solution) = 0;

        virtual ErrorBase *
        compute_time_error(std::vector<ErrorBase *> &errors) = 0;

        double
        compute_condition_number();


        virtual void
        write_time_header_to_file(std::ofstream &file) = 0;

        virtual void
        write_time_error_to_file(ErrorBase *error, std::ofstream &file) = 0;


        virtual void
        output_results(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                       LA::MPI::Vector &solution,
                       std::string &suffix,
                       bool minimal_output = false) const;

        virtual void
        output_results(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                       LA::MPI::Vector &solution,
                       int time_step,
                       bool minimal_output = false) const;

        virtual void
        output_results(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                       LA::MPI::Vector &solution,
                       bool minimal_output = false) const;

        void
        output_levelset(int time_step) const;

        virtual void
        post_processing(unsigned int time_step);
        

        MPI_Comm mpi_communicator;

        parallel::distributed::Triangulation<dim> triangulation;

        const unsigned int n_refines;
        const unsigned int element_order;
        bool write_output;

        double h;  // cell side length
        double tau;

        hp::FECollection<dim> fe_collection;
        hp::MappingCollection<dim> mapping_collection;
        hp::QCollection<dim> q_collection;
        hp::QCollection<1> q_collection1D;

        // Object managing degrees of freedom for the level set function.
        FE_Q<dim> fe_levelset;
        DoFHandler<dim> levelset_dof_handler;
        LA::MPI::Vector levelset;
        IndexSet ls_locally_owned_dofs;
        IndexSet ls_locally_relevant_dofs;

        LevelSet<dim> *levelset_function;
        bool moving_domain = false;

        // Object managing degrees of freedom for the cutfem method.
        std::deque<std::shared_ptr<hp::DoFHandler<dim>>> dof_handlers;

        // TODO do we need a deque for each of these too??
        //  - I dont think this can be done. Any processor can only know about
        //    it's own dofs, so if these changed for each time step, we would 
        //    not be able to perform computations using the previous time step.
        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;

        NonMatching::MeshClassifier<dim> cut_mesh_classifier;

        LA::MPI::SparseMatrix stiffness_matrix;
        LA::MPI::SparseMatrix timedep_stiffness_matrix;
        LA::MPI::Vector rhs;

        AffineConstraints<double> constraints;

        // Queue of current and  previous solutions, used in the time
        // discretization method. When a new time step is solved, a new empty
        // solution vector is pushed to the front.
        //  - In the first iteration of BDF-2 it containts (u, u1, u0).
        //  - This is only the locally relevant solution, when the program is 
        //    run in parallell.
        std::deque<LA::MPI::Vector> solutions;

        // Constants used for the time discretization, defined as:
        //   u_t = (au^(n+1) + bu^n + cu^(n-1))/τ, where u = u^(n+1)
        // For BDF-1: (a, b), and (a, b, c) for BDF-2.
        std::vector<double> bdf_coeffs;

        // Extrapolation coefficients, used for the discretisation of the
        // convection term for Navier-Stokes.
        std::vector<double> extrap_coeffs;

        bool crank_nicholson;

        const bool stabilized;

        // Set to true to solve the time dependent problem.
        const bool stationary;

        // Set to false to skip the error computations.
        const bool do_compute_error;

        // When this flag is set to false, it is assumed we are solving a
        // non linear problem, so the non-linear part of the stiffness matrix
        // has to be assembled again in each time step, even if the domain is
        // stationary. The stationary part of the stiffness matrix is denoted
        // by A, and is assembled by assemble_matrix(), while the non linearized
        // part C(u_e) is assembled by assemble_timedep_matrix(), where u_e
        // is e.g. the extrapolated solution. The solve() method then solves
        // the system (A + C(u_e))u = f instead. This is done when the
        // Navier-Stokes equations are solved with a semi-implicit convection
        // term.
        bool stationary_stiffness_matrix = true;

        ConditionalOStream pcout;
        TimerOutput computing_timer;

        const unsigned int n_mpi_processes;
        const unsigned int this_mpi_process;
    };
}

#endif //MICROBUBBLE_CUTFEM_PROBLEM_H
