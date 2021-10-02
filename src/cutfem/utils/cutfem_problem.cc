#include <deal.II/base/data_out_base.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/non_matching/fe_values.h>

#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>


#include "cutfem_problem.h"


namespace utils::problems {

    template<int dim>
    CutFEMProblem<dim>::
    CutFEMProblem(const unsigned int n_refines,
                  const int element_order,
                  const bool write_output,
                  Function<dim> &levelset_func,
                  const bool stabilized)
            : n_refines(n_refines), element_order(element_order),
              write_output(write_output),
              fe_levelset(element_order),
              levelset_dof_handler(triangulation),
              cut_mesh_classifier(triangulation, levelset_dof_handler,
                                  levelset),
              stabilized(stabilized) {
        // Use no constraints when projecting.
        this->constraints.close();

        levelset_function = &levelset_func;
    }


    template<int dim>
    ErrorBase *CutFEMProblem<dim>::
    run_step() {
        make_grid(triangulation);
        setup_quadrature();
        setup_level_set();
        cut_mesh_classifier.reclassify(); // TODO move this into distribute_dofs method
        dof_handlers.emplace_front(triangulation);
        setup_fe_collection();
        distribute_dofs(dof_handlers.front());
        initialize_matrices();
        int n_dofs = dof_handlers.front().n_dofs();
        solutions.emplace_front(n_dofs);
        this->assemble_system();
        solve();
        if (write_output) {
            output_results();
        }
        return compute_error(dof_handlers.front(), solutions.front());
    }


    template<int dim>
    Vector<double> CutFEMProblem<dim>::
    get_solution() {
        return solutions.front();
    }


    template<int dim>
    hp::DoFHandler<dim> &CutFEMProblem<dim>::
    get_dof_handler() {
        // TODO return reference or pointer?
        return dof_handlers.front();
    }


    template<int dim>
    ErrorBase *CutFEMProblem<dim>::
    run_time(unsigned int bdf_type, unsigned int steps,
             std::vector<Vector<double>> &supplied_solutions) {

        std::cout << "\nBDF-" << bdf_type << ", steps=" << steps << std::endl;

        make_grid(this->triangulation);
        setup_quadrature();
        setup_level_set();
        cut_mesh_classifier.reclassify();
        setup_fe_collection();
        // Initialize the first dof_handler.
        dof_handlers.emplace_front(triangulation);
        distribute_dofs(dof_handlers.front());
        initialize_matrices();

        // Vector for the computed error for each time step.
        std::vector<ErrorBase *> errors(steps + 1);

        interpolate_first_steps(bdf_type, errors);

        std::vector<std::reference_wrapper<hp::DoFHandler<dim>>> no_dof_handlers;
        set_supplied_solutions(bdf_type, supplied_solutions,
                               no_dof_handlers, errors);
        set_bdf_coefficients(bdf_type);

        // TODO note that the next solution vector is not yet set in solutions,
        //  this may lead to problems. Eg for Crank-Nicholson?
        assemble_matrix();

        std::ofstream file("errors-time-d" + std::to_string(dim)
                           + "o" + std::to_string(this->element_order)
                           + "r" + std::to_string(this->n_refines) + ".csv");
        write_time_header_to_file(file);

        // Write the errors for the first steps to file.
        for (unsigned int k = 0; k < bdf_type; ++k) {
            write_time_error_to_file(errors[k], file);
            errors[k]->output();
        }

        double time;
        for (unsigned int k = bdf_type; k <= steps; ++k) {
            time = k * this->tau;
            std::cout << "\nTime Step = " << k
                      << ", tau = " << this->tau
                      << ", time = " << time << std::endl;

            // Advance the time for all functions.
            set_function_times(time);

            // Create a new solution vector to contain the next solution.
            int n_dofs = this->dof_handlers.front().n_dofs();
            this->solutions.emplace_front(n_dofs);

            // TODO nødvendig??
            this->rhs.reinit(this->solutions.front().size());

            assemble_rhs(k, false);
            this->solve();
            errors[k] = this->compute_error(dof_handlers.front(),
                                            solutions.front());
            errors[k]->time_step = k;
            write_time_error_to_file(errors[k], file);
            errors[k]->output();

            if (this->write_output) {
                this->output_results(k, true);
            }

            // Remove the oldest solution, since it is no longer needed.
            solutions.pop_back();
        }

        std::cout << std::endl;
        for (ErrorBase *error : errors) {
            error->output();
        }

        return compute_time_error(errors);
    }


    template<int dim>
    ErrorBase *CutFEMProblem<dim>::
    run_time(unsigned int bdf_type, unsigned int steps) {
        // Invoking this method will result in a pure BDF-k method, where all
        // the initial steps will be interpolated.
        std::vector<Vector<double>> empty;
        return run_time(bdf_type, steps, empty);
    }


    template<int dim>
    ErrorBase *CutFEMProblem<dim>::
    run_moving_domain(unsigned int bdf_type, unsigned int steps,
                      std::vector<Vector<double>> &supplied_solutions,
                      std::vector<std::reference_wrapper<hp::DoFHandler<dim>>> &supplied_dof_handlers) {

        std::cout << "\nBDF-" << bdf_type << ", steps=" << steps << std::endl;

        // One dof_handler must be supplied for each supplied solution vector.
        assert(supplied_solutions.size() == supplied_dof_handlers.size());

        make_grid(this->triangulation);
        setup_quadrature();
        setup_level_set();
        cut_mesh_classifier.reclassify(); // TODO any reason to keep this call outside the method above?
        setup_fe_collection();

        // dof_handlers = std::vector<hp::DoFHandler<dim>(bdf_type);
        dof_handlers.emplace_front(triangulation);
        distribute_dofs(dof_handlers.front());
        initialize_matrices();

        // Vector for the computed error for each time step.
        std::vector<ErrorBase *> errors(steps + 1);

        interpolate_first_steps(bdf_type, errors, true);
        set_supplied_solutions(bdf_type, supplied_solutions,
                               supplied_dof_handlers, errors, true);
        set_bdf_coefficients(bdf_type);

        std::ofstream file("errors-time-d" + std::to_string(dim)
                           + "o" + std::to_string(element_order)
                           + "r" + std::to_string(n_refines) + ".csv");
        write_time_header_to_file(file);

        // Write the errors for the first steps to file.
        for (unsigned int k = 0; k < bdf_type; ++k) {
            write_time_error_to_file(errors[k], file);
            errors[k]->output();
        }

        // Check that we have created exactly one dof_handler per solution.
        assert(dof_handlers.size() == solutions.size());

        double time;
        for (unsigned int k = bdf_type; k <= steps; ++k) {
            time = k * tau;
            std::cout << "\nTime Step = " << k
                      << ", tau = " << tau
                      << ", time = " << time << std::endl;

            set_function_times(time);
            setup_level_set();
            cut_mesh_classifier.reclassify(); // TODO kalles denne i riktig rekkefølge?

            // Create a new solution vector to contain the next solution.
            int n_dofs = dof_handlers.front().n_dofs();
            solutions.emplace_front(n_dofs);

            // Redistribute the dofs after the level set was updated
            dof_handlers.emplace_front(triangulation);
            distribute_dofs(dof_handlers.front());
            // Reinitialize the matrices and vectors after the number of dofs
            // was updated.
            initialize_matrices();

            assemble_matrix();
            assemble_rhs(k, true);

            solve();
            errors[k] = compute_error(dof_handlers.front(), solutions.front());
            errors[k]->time_step = k;
            write_time_error_to_file(errors[k], file);
            errors[k]->output();

            if (write_output) {
                output_results(k, false);
            }

            // Remove the oldest solution and dof_handler, since they are
            // no longer needed.
            solutions.pop_back();
            dof_handlers.pop_back();
        }

        std::cout << std::endl;
        for (ErrorBase *error : errors) {
            error->output();
        }
        return compute_time_error(errors);
    }


    template<int dim>
    ErrorBase *CutFEMProblem<dim>::
    run_moving_domain(unsigned int bdf_type, unsigned int steps) {
        // Invoking this method will result in a pure BDF-k method, where all
        // the initial steps will be interpolated.
        // TODO store the solutions as references too
        std::vector<Vector<double>> empty_solutions;
        std::vector<std::reference_wrapper<hp::DoFHandler<dim>>> empty_dof_h;
        return run_moving_domain(bdf_type, steps, empty_solutions, empty_dof_h);
    }


    template<int dim>
    void CutFEMProblem<dim>::
    set_bdf_coefficients(unsigned int bdf_type) {
        std::cout << "Set BDF coefficients" << std::endl;
        this->bdf_coeffs = std::vector<double>(bdf_type + 1);

        if (bdf_type == 1) {
            // BDF-1 (implicit Euler).
            bdf_coeffs[0] = 1;
            bdf_coeffs[1] = -1;
        } else if (bdf_type == 2) {
            // BDF-2.
            bdf_coeffs[0] = 1.5;
            bdf_coeffs[1] = -2;
            bdf_coeffs[2] = 0.5;
        } else if (bdf_type == 3) {
            bdf_coeffs[0] = 11.0 / 6;
            bdf_coeffs[1] = -3;
            bdf_coeffs[2] = 1.5;
            bdf_coeffs[3] = -1.0 / 3;
        } else {
            throw std::invalid_argument(
                    "bdf_type has to be either 1 or 2, not " +
                    std::to_string(bdf_type) + ".");
        }
    }

    /**
     *
     * @tparam dim
     * @param errors
     * @param u1
     * @param bdf_type
     *
     * The initial value u0 is interpolated using the boundary_values object,
     * while possibly next steps are interpolated using analytical_velocity and
     * analytical_pressure. This is becuase this is only done when u1 is not
     * supplied, which must mean we have the analytical solutions.
     *
     */
    template<int dim>
    void CutFEMProblem<dim>::
    interpolate_first_steps(unsigned int bdf_type,
                            std::vector<ErrorBase *> &errors,
                            bool moving_domain) {
        std::cout << "Interpolate first step(s)." << std::endl;
        // Assume the deque of solutions is empty at this point.
        assert(solutions.size() == 0);
        // At this point, one dof_handler should have been created.
        assert(dof_handlers.size() == 1);

        int n_dofs = dof_handlers.front().n_dofs();

        for (unsigned int k = 0; k < bdf_type; ++k) {
            // Create a new solution vector.
            solutions.emplace_front(n_dofs);
            if (moving_domain && k > 0) {
                // For moving domains we need a new dof_handler for each step,
                // but the first one should already have been created.
                dof_handlers.emplace_front(triangulation);
                distribute_dofs(dof_handlers.front());
            }

            // Interpolate it a the correct time.
            set_function_times(k * this->tau);
            interpolate_solution(dof_handlers.front(), k, moving_domain);
            // TODO create dofhandler before interpolating

            // Compute the error for this step.
            errors[k] = this->compute_error(dof_handlers.front(),
                                            solutions.front());
            errors[k]->time_step = k;
            std::string suffix = std::to_string(k) + "-inter";
            this->output_results(suffix);
        }

        // TODO burde kanskje heller interpolere boundary_values for
        //  initial verdier?
        // Important that the boundary_values function uses t=0, when
        // we interpolate the initial value from it.
        // boundary_values->set_time(0);

        // Use the boundary_values as initial values. Interpolate the
        // boundary_values function into the finite element space.
        // const unsigned int n_components_on_element = dim + 1;
        // FEValuesExtractors::Vector velocities(0);
        //VectorFunctionFromTensorFunction<dim> adapter(
        //        *boundary_values,
        //        velocities.first_vector_component,
        //        n_components_on_element);

    }

    template<int dim>
    void CutFEMProblem<dim>::
    set_supplied_solutions(unsigned int bdf_type,
                           std::vector<Vector<double>> &supplied_solutions,
                           std::vector<std::reference_wrapper<hp::DoFHandler<dim>>> &supplied_dof_handlers,
                           std::vector<ErrorBase *> &errors,
                           bool moving_domain) {
        std::cout << "Set supplied solutions" << std::endl;
        std::cout << "  solutions.size() = " << solutions.size() << std::endl;

        // At this point we assume the solution vector that are used for solving
        // the next time step is not yet created.
        assert(solutions.size() == bdf_type);

        if (!moving_domain) {
            // In cases with a stationary domain, a dof_handler is created and
            // accessed with dof_handler.front(), so this is used for all further
            // time steps.
            assert(supplied_dof_handlers.size() == 0);
            // For stationary domains, we only need one dof_handler.
            assert(dof_handlers.size() == 1);
        }

        // Create an extended vector of supplied_solutions, with vectors of
        // length 1 to mark the time steps where we want to keep and use the
        // interpolated solution.
        std::vector<Vector<double>> full_vector(bdf_type, Vector<double>(1));
        unsigned int num_supp = supplied_solutions.size();
        unsigned int size_diff = bdf_type - num_supp;
        assert(size_diff >= 0);
        for (unsigned int k = 0; k < num_supp; ++k) {
            full_vector[size_diff + k] = supplied_solutions[k];
        }

        // Insert the supplied solutions in the solutions deque, and compute
        // the errors.
        unsigned int solution_index;
        unsigned int n_dofs = this->dof_handlers.front().n_dofs();
        for (unsigned int k = 0; k < bdf_type; ++k) {
            if (full_vector[k].size() == n_dofs) {
                std::cout << " - Set supplied for k = " << k << std::endl;
                // Flip the index, since the solutions deque and the
                // supplied solutions vector holds the solution vectors
                // in opposite order.
                solution_index = solutions.size() - 1 - k;
                // Overwrite the interpolated solution for this step, since it
                // was supplied to the solver.
                solutions[solution_index] = full_vector[k];
                set_function_times(k * this->tau);
                // Overwrite the error too.
                // TODO tilpass til moving_domain er true
                errors[k] = this->compute_error(this->dof_handlers.front(),
                                                solutions[solution_index]);
                errors[k]->time_step = k;
            }
        }
    }


    template<int dim>
    void CutFEMProblem<dim>::
    set_function_times(double time) {
        throw std::logic_error(
                "Override this method to run a time dependent problem.");
    }


    template<int dim>
    void CutFEMProblem<dim>::
    interpolate_solution(hp::DoFHandler<dim> &dof_handler, int time_step,
                         bool moving_domain) {
        throw std::logic_error(
                "Override this method to run a time dependent problem.");
    }


    template<int dim>
    void CutFEMProblem<dim>::
    setup_quadrature() {
        const unsigned int quadOrder = 2 * element_order + 1;
        q_collection.push_back(QGauss<dim>(quadOrder));
        q_collection1D.push_back(QGauss<1>(quadOrder));
    }


    template<int dim>
    void CutFEMProblem<dim>::
    setup_level_set() {
        std::cout << "Setting up level set" << std::endl;

        // The level set function lives on the whole background mesh.
        levelset_dof_handler.distribute_dofs(fe_levelset);
        printf("  leveset dofs: %d\n", levelset_dof_handler.n_dofs());
        levelset.reinit(levelset_dof_handler.n_dofs());

        // Project the geometry onto the mesh.
        VectorTools::project(levelset_dof_handler,
                             constraints,
                             QGauss<dim>(2 * element_order + 1),
                             *levelset_function,
                             levelset);
    }


    template<int dim>
    void CutFEMProblem<dim>::
    initialize_matrices() {
        std::cout << "Initialize marices" << std::endl;
        int n_dofs = dof_handlers.front().n_dofs();
        rhs.reinit(n_dofs);

        cutfem::nla::make_sparsity_pattern_for_stabilized(dof_handlers.front(),
                                                          sparsity_pattern);
        stiffness_matrix.reinit(sparsity_pattern);
    }


    template<int dim>
    void CutFEMProblem<dim>::
    assemble_system() {}

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_local_over_cell(const FEValues<dim> &fe_values,
                             const std::vector<types::global_dof_index> &loc2glb) {}

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_local_over_surface(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {}


    template<int dim>
    void CutFEMProblem<dim>::
    assemble_matrix() {}

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_matrix_local_over_cell(const FEValues<dim> &fe_values,
                                    const std::vector<types::global_dof_index> &loc2glb) {}

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_matrix_local_over_surface(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {}

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_rhs(int time_step, bool moving_domain) {}

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_rhs_local_over_cell(const FEValues<dim> &fe_values,
                                 const std::vector<types::global_dof_index> &loc2glb) {}

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_rhs_local_over_cell_cn(const FEValues<dim> &fe_values,
                                    const std::vector<types::global_dof_index> &loc2glb,
                                    const int time_step) {}

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_rhs_local_over_surface(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glob) {}

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_rhs_local_over_surface_cn(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glob,
            const int time_step) {}


    template<int dim>
    void CutFEMProblem<dim>::
    solve() {
        std::cout << "Solving system" << std::endl;
        SparseDirectUMFPACK inverse;
        inverse.initialize(stiffness_matrix);
        inverse.vmult(solutions.front(), rhs);
    }


    template<int dim>
    double CutFEMProblem<dim>::
    compute_condition_number() {
        std::cout << "Compute condition number" << std::endl;

        // Invert the stiffness_matrix
        FullMatrix<double> stiffness_matrix_full(solutions.front().size());
        stiffness_matrix_full.copy_from(stiffness_matrix);
        FullMatrix<double> inverse(solutions.front().size());
        inverse.invert(stiffness_matrix_full);

        double norm = stiffness_matrix.frobenius_norm();
        double inverse_norm = inverse.frobenius_norm();

        double condition_number = norm * inverse_norm;
        std::cout << "  cond_num = " << condition_number << std::endl;

        // TODO bruk eigenvalues istedet
        return condition_number;
    }


    template<int dim>
    void CutFEMProblem<dim>::
    output_results(bool minimal_output) const {
        std::string empty;
        output_results(empty, minimal_output);
    }


    template<int dim>
    void CutFEMProblem<dim>::
    output_results(int time_step, bool minimal_output) const {
        std::string k = std::to_string(time_step);
        output_results(k, minimal_output);
    }


    template
    class CutFEMProblem<2>;

    template
    class CutFEMProblem<3>;

}

