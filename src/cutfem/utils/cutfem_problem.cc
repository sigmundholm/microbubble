#include <deal.II/base/data_out_base.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_update_flags.h>

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

#include "utils.h"
#include "cutfem_problem.h"


namespace utils::problems {

    template<int dim>
    Tensor<1, dim> LevelSet<dim>::
    get_velocity() {
        Tensor<1, dim> zero;
        return zero;
    }

    template<int dim>
    double LevelSet<dim>::
    get_speed() {
        return sqrt(get_velocity().norm_square());
    }


    template<int dim>
    CutFEMProblem<dim>::
    CutFEMProblem(const unsigned int n_refines,
                  const int element_order,
                  const bool write_output,
                  LevelSet<dim> &levelset_func,
                  const bool stabilized,
                  const bool stationary,
                  const bool compute_error)
            : mpi_communicator(MPI_COMM_WORLD), 
              triangulation(mpi_communicator,
                            typename Triangulation<dim>::MeshSmoothing(
                              Triangulation<dim>::smoothing_on_refinement |
                              Triangulation<dim>::smoothing_on_coarsening)),
              n_refines(n_refines), element_order(element_order),
              write_output(write_output),
              fe_levelset(element_order),
              levelset_dof_handler(triangulation),
              cut_mesh_classifier(levelset_dof_handler, levelset),
              stabilized(stabilized), stationary(stationary),
              do_compute_error(compute_error), 
              pcout(std::cout, 
                    (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)), 
              computing_timer(mpi_communicator, 
                              pcout, 
                              TimerOutput::never, 
                              TimerOutput::wall_times),
              n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
              this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)) {
        // Use no constraints when projecting.
        this->constraints.close();

        levelset_function = &levelset_func;
    }


    template<int dim>
    ErrorBase *CutFEMProblem<dim>::
    run_step() {
        pcout << "Solve equation: stationary." << std::endl;
        pcout << "---------------------------\n" << std::endl;
        pcout << "Running with "
#ifdef USE_PETSC_LA
              << "PETSc"
#else
              << "Trilinos"
#endif
              << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
              << " MPI rank(s)." << std::endl;

        make_grid(triangulation);
        std::cout << "  n_cells = " << triangulation.n_cells() << std::endl;
        set_grid_size();
        setup_quadrature();
        set_function_times(0);
        setup_level_set();
        cut_mesh_classifier.reclassify(); // TODO move this into distribute_dofs method
        dof_handlers.emplace_front(new hp::DoFHandler<dim>(triangulation));
        setup_fe_collection();
        distribute_dofs(dof_handlers.front());

        // Get the dofs owned by this processor. This should probably be done 
        // only once during each run, to make sure they dont differ across time
        // steps.
        locally_owned_dofs = dof_handlers.front()->locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(*dof_handlers.front(), 
                                                locally_relevant_dofs);

        set_bdf_coefficients(1);
        set_extrapolation_coefficients(1);
        
        solutions.emplace_front(locally_owned_dofs, locally_relevant_dofs,
                                mpi_communicator);

        initialize_stiffness_matrix();
        pre_matrix_assembly();
        {
            TimerOutput::Scope t(computing_timer, "assembly");
            assemble_system();
        }

        solve();
        post_processing(0);

        if (write_output) {
            TimerOutput::Scope t(computing_timer, "output");
            output_results(this->dof_handlers.front(),
                           this->solutions.front());
        }

        ErrorBase* error;
        if (do_compute_error) {
            TimerOutput::Scope t(computing_timer, "compute error");
            error = compute_error(dof_handlers.front(), solutions.front());
        }
        computing_timer.print_summary();
        computing_timer.reset();
        return error;
    }


    template<int dim>
    ErrorBase *CutFEMProblem<dim>::
    run_step_non_linear(double tol) {
        pcout << "Solve equation: non-linear." << std::endl;
        pcout << "---------------------------\n" << std::endl;

        make_grid(triangulation);
        std::cout << "  n_cells = " << triangulation.n_cells() << std::endl;
        set_grid_size();
        setup_quadrature();
        set_function_times(0);
        setup_level_set();
        cut_mesh_classifier.reclassify(); // TODO move this into distribute_dofs method
        dof_handlers.emplace_front(new hp::DoFHandler<dim>(triangulation));
        setup_fe_collection();
        distribute_dofs(dof_handlers.front());

        // Get the dofs owned by this processor. This should probably be done 
        // only once during each run, to make sure they dont differ across time
        // steps.
        locally_owned_dofs = dof_handlers.front()->locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(*dof_handlers.front(), 
                                                locally_relevant_dofs);

        set_bdf_coefficients(1);
        set_extrapolation_coefficients(1);

        double prev_error;
        double this_error = 1; // Set to 1 to enforce at least two steps.
        ErrorBase *error;
        double error_diff = 2 * tol;
        int k = 0;

        solutions.emplace_front(locally_owned_dofs, locally_relevant_dofs, 
                                mpi_communicator);

        while (error_diff > tol) {
            k++;
            pcout << "\nFixed point iteration: step " << k << std::endl;
            pcout << "-----------------------------------" << std::endl;

            solutions.emplace_front(locally_owned_dofs, locally_relevant_dofs, 
                                    mpi_communicator);
            if (k == 1) {
                initialize_stiffness_matrix();
                pre_matrix_assembly();
                assemble_matrix();
            }
            if (!stationary_stiffness_matrix) {
                initialize_timedep_matrix();
                assemble_timedep_matrix();
            }

            rhs = 0;
            assemble_rhs(0);

            solve();
            post_processing(k);

            error = compute_error(dof_handlers.front(), solutions.front());
            prev_error = this_error;
            this_error = error->repr_error();
            error->output();
            error_diff = abs(this_error - prev_error);
            pcout << "  Error diff = " << error_diff << std::endl;

            if (write_output) {
                output_results(dof_handlers.front(), solutions.front(),
                               k, k > 1);
            }
            solutions.pop_back();
        }
        post_processing(k + 1);

        if (do_compute_error) {
            return compute_error(dof_handlers.front(), solutions.front());
        } else {
            return nullptr;
        }
    }


    template<int dim>
    LA::MPI::Vector CutFEMProblem<dim>::
    get_solution() {
        return solutions.front();
    }


    template<int dim>
    std::shared_ptr<hp::DoFHandler<dim>> CutFEMProblem<dim>::
    get_dof_handler() {
        return dof_handlers.front();
    }


    template<int dim>
    ErrorBase *CutFEMProblem<dim>::
    run_time(unsigned int bdf_type, unsigned int steps,
             std::vector<LA::MPI::Vector> &supplied_solutions) {
        pcout << "Solve equation: time-depenedent." << std::endl;
        pcout << "--------------------------------" << std::endl;
        pcout << "BDF-" << bdf_type << ", steps=" << steps << std::endl;
        pcout << "-------------------------" << std::endl;
        pcout << "Running with "
#ifdef USE_PETSC_LA
              << "PETSc"
#else
              << "Trilinos"
#endif
              << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
              << " MPI rank(s)." << std::endl;

        assert(supplied_solutions.size() < bdf_type);
        // Clear the solutions and dof_handlers from possibly previous BDF
        // method runs performed by this object.
        solutions.clear();
        dof_handlers.clear();

        // Don't make the triangulation if it was done by a previously run
        // of a BDF-method.
        if (triangulation.n_quads() == 0) {
            make_grid(triangulation);
            std::cout << "  n_cells = " << triangulation.n_cells() << std::endl;
            set_grid_size();
            setup_quadrature();
        }
        set_function_times(0);
        setup_level_set();
        cut_mesh_classifier.reclassify();
        setup_fe_collection();
        // Initialize the first dof_handler.
        dof_handlers.emplace_front(new hp::DoFHandler<dim>());
        distribute_dofs(dof_handlers.front());
        
        // Get the dofs owned by this processor. This should probably be done 
        // only once during each run, to make sure they dont differ across time
        // steps.
        // TODO what happens for BDF-2?
        locally_owned_dofs = dof_handlers.front()->locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(*dof_handlers.front(), 
                                                locally_relevant_dofs);

        // Vector for the computed error for each time step.
        std::vector<ErrorBase *> errors(steps + 1);

        interpolate_first_steps(bdf_type, errors);

        // When the domain is stationary, we dont need to supply any DoFHandlers.
        std::vector<std::shared_ptr<hp::DoFHandler<dim>>> no_dof_handlers;
        set_supplied_solutions(bdf_type, supplied_solutions,
                               no_dof_handlers, errors);
        set_bdf_coefficients(bdf_type);
        set_extrapolation_coefficients(bdf_type);

        std::ofstream file("errors-time-d" + std::to_string(dim)
                           + "o" + std::to_string(element_order)
                           + "r" + std::to_string(n_refines) + ".csv");
        write_time_header_to_file(file);

        // Write the errors for the first steps to file.
        for (unsigned int k = 0; k < bdf_type; ++k) {
            write_time_error_to_file(errors[k], file);
            errors[k]->output();
        }

        double time;
        for (unsigned int k = bdf_type; k <= steps; ++k) {
            time = k * tau;
            pcout << "\nTime Step = " << k
                      << ", tau = " << tau
                      << ", time = " << time << std::endl;

            // Advance the time for all functions.
            set_function_times(time);

            // Create a new solution vector to contain the next solution.
            solutions.emplace_front(locally_owned_dofs, locally_relevant_dofs, 
                                    mpi_communicator);

            if (k == bdf_type) {
                // Assemble the matrix after the new solution vector is created.
                // This is to omit index problems when assembling a stiffness
                // matrix that is dependent on previous solutions.

                // Assemble the stiffness matrix
                initialize_stiffness_matrix();
                pre_matrix_assembly();
                assemble_matrix();
            }
            if (!stationary_stiffness_matrix) {
                initialize_timedep_matrix();
                assemble_timedep_matrix();
            }

            rhs = 0;
            assemble_rhs(k);

            solve();
            post_processing(k);

            if (do_compute_error) {
                // TODO segfault when this is compute_error = false.
                errors[k] = compute_error(dof_handlers.front(),
                                          solutions.front());
                errors[k]->time_step = k;
                write_time_error_to_file(errors[k], file);
                errors[k]->output();
            }

            if (write_output) {
                output_results(dof_handlers.front(), solutions.front(),
                               k, true);
            }

            // Remove the oldest solution, since it is no longer needed.
            solutions.pop_back();
        }

        pcout << std::endl;
        for (ErrorBase *error : errors) {
            error->output();
        }
        if (do_compute_error) {
            return compute_time_error(errors);
        } else {
            return nullptr;
        }
    }


    template<int dim>
    ErrorBase *CutFEMProblem<dim>::
    run_time(unsigned int bdf_type, unsigned int steps) {
        // Invoking this method will result in a pure BDF-k method, where all
        // the initial steps will be interpolated.
        std::vector<LA::MPI::Vector> empty;
        return run_time(bdf_type, steps, empty);
    }


    template<int dim>
    ErrorBase *CutFEMProblem<dim>::
    run_moving_domain(unsigned int bdf_type, unsigned int steps,
                      std::vector<LA::MPI::Vector> &supplied_solutions,
                      std::vector<std::shared_ptr<hp::DoFHandler<dim>>> &supplied_dof_handlers,
                      const double mesh_bound_multiplier) {

        pcout << "\nBDF-" << bdf_type << ", steps=" << steps << std::endl;
        pcout << "-------------------------" << std::endl;
        moving_domain = true;

        // One dof_handler must be supplied for each supplied solution vector.
        assert(supplied_solutions.size() == supplied_dof_handlers.size());
        // Clear the solutions and dof_handlers from possibly previous BDF
        // method runs performed by this object.
        solutions.clear();
        dof_handlers.clear();

        // Don't make the triangulation if it was done by a previously run
        // of a BDF-method.
        if (triangulation.n_quads() == 0) {
            make_grid(triangulation);
            std::cout << "  n_cells = " << triangulation.n_cells() << std::endl;
            set_grid_size();
            setup_quadrature();
        }
        set_function_times(0);
        setup_level_set();
        cut_mesh_classifier.reclassify(); // TODO any reason to keep this call outside the method above?
        setup_fe_collection();

        // TODO compute the speed at each cell, to get a more precise calculation.

        // Note that when using BDF-2 with BDF-1 for the u1 step, the acitve
        // mesh for the BDF-1 method should be enlarged with a factor 2 with
        // the use of the mesh_bound_multiplier, else the mesh will be two
        // small when solving the step k=3 with BDF-2. This is naturally because
        // of the constant bdf_type is used in the size_of_bound constant.
        double buffer_constant = 1.5;
        double size_of_bound = mesh_bound_multiplier * buffer_constant
                               * (levelset_function->get_speed() * tau
                                  * bdf_type + h);
        pcout << " # size_of_bound = " << size_of_bound << std::endl;

        dof_handlers.emplace_front(new hp::DoFHandler<dim>());
        distribute_dofs(dof_handlers.front(), size_of_bound);
        
        // Get the dofs owned by this processor. This should probably be done 
        // only once during each run, to make sure they dont differ across time
        // steps.
        locally_owned_dofs = dof_handlers.front()->locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(*dof_handlers.front(), 
                                                locally_relevant_dofs);

        initialize_stiffness_matrix();

        // Vector for the computed error for each time step.
        std::vector<ErrorBase *> errors(steps + 1);

        interpolate_first_steps(bdf_type, errors, mesh_bound_multiplier);
        set_supplied_solutions(bdf_type, supplied_solutions,
                               supplied_dof_handlers, errors);
        set_bdf_coefficients(bdf_type);
        set_extrapolation_coefficients(bdf_type);

        std::ofstream file("errors-time-d" + std::to_string(dim)
                           + "o" + std::to_string(element_order)
                           + "r" + std::to_string(n_refines) + ".csv");
        write_time_header_to_file(file);

        pcout << "Interpolated / supplied solutions." << std::endl;
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
            pcout << "\nTime Step = " << k
                      << ", tau = " << tau
                      << ", time = " << time << std::endl;

            set_function_times(time);
            setup_level_set();
            cut_mesh_classifier.reclassify(); // TODO kalles denne i riktig rekkefølge?

            // Create a new solution vector to contain the next solution.
            solutions.emplace_front(locally_owned_dofs, locally_relevant_dofs, 
                                    mpi_communicator);

            // Redistribute the dofs after the level set was updated
            // size_of_bound = buffer_constant * bdf_type * this->h;
            size_of_bound = mesh_bound_multiplier * buffer_constant
                            * (levelset_function->get_speed() * tau
                               * bdf_type + h);
            pcout << " # size_of_bound = " << size_of_bound << std::endl;
            dof_handlers.emplace_front(new hp::DoFHandler<dim>());
            distribute_dofs(dof_handlers.front(), size_of_bound);

            // Reinitialize the matrices and vectors after the number of dofs
            // was updated.
            initialize_stiffness_matrix();

            pre_matrix_assembly();
            assemble_matrix();

            // Note since the domain is moving, the whole stiffness matrix has
            // to be assembled each time step.
            if (!stationary_stiffness_matrix) 
                assemble_timedep_matrix();
            assemble_rhs(k);

            solve();
            post_processing(k);

            if (do_compute_error) {
                errors[k] = compute_error(dof_handlers.front(),
                                          solutions.front());
                errors[k]->time_step = k;
                write_time_error_to_file(errors[k], file);
                errors[k]->output();
            }

            if (write_output) {
                output_results(this->dof_handlers.front(),
                               this->solutions.front(), k, false);
            }

            // Remove the oldest solution and dof_handler, since they are
            // no longer needed.
            solutions.pop_back();
            dof_handlers.pop_back();
        }

        pcout << std::endl;
        for (ErrorBase *error : errors) {
            error->output();
        }
        if (do_compute_error) {
            return compute_time_error(errors);
        } else {
            return nullptr;
        }
    }


    template<int dim>
    ErrorBase *CutFEMProblem<dim>::
    run_moving_domain(unsigned int bdf_type, unsigned int steps,
                      const double mesh_bound_multiplier) {
        // Invoking this method will result in a pure BDF-k method, where all
        // the initial steps will be interpolated.
        std::vector<LA::MPI::Vector> empty_solutions;
        std::vector<std::shared_ptr<hp::DoFHandler<dim>>> empty_dof_h;
        return run_moving_domain(bdf_type, steps, empty_solutions, empty_dof_h,
                                 mesh_bound_multiplier);
    }


    template<int dim>
    void CutFEMProblem<dim>::
    set_bdf_coefficients(unsigned int bdf_type) {
        pcout << "Set BDF coefficients" << std::endl;
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


    template<int dim>
    void CutFEMProblem<dim>::
    set_extrapolation_coefficients(unsigned int bdf_type) {
        // Initialize the vector with extrapolation coefficients.
        extrap_coeffs = std::vector<double>(bdf_type + 1);
        switch (bdf_type) {
            case 1:
                // 1st order extrapolation.
                extrap_coeffs[0] = 0;
                extrap_coeffs[1] = 1;
                break;
            case 2:
                // 2nd order extrapolation.
                extrap_coeffs[0] = 0;
                extrap_coeffs[1] = 2;
                extrap_coeffs[2] = -1;
                break;
            case 3:
                // 3rd order extrapolation.
                extrap_coeffs[0] = 0;
                extrap_coeffs[1] = 3;
                extrap_coeffs[2] = -3;
                extrap_coeffs[3] = 1;
                break;
            default:
                throw std::invalid_argument(
                        "Extrapolation of order k = "
                        + std::to_string(this->solutions.size() - 1)
                        + ", is not implemented.");
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
                            double mesh_bound_multiplier) {
        pcout << "Interpolate first step(s)." << std::endl;
        // Assume the deque of solutions is empty at this point.
        assert(solutions.empty());
        // At this point, one dof_handler should have been created.
        assert(dof_handlers.size() == 1);

        for (unsigned int k = 0; k < bdf_type; ++k) {
            // Create a new solution vector.
            pcout << " - Interpolate step k = " << k << std::endl;

            // Interpolate it a the correct time.
            set_function_times(k * tau);

            if (moving_domain && k > 0) {
                // For moving domains we need a new dof_handler for each step,
                // but the first one should already have been created.
                double buffer_constant = 1.5;
                // TODO take this calculation out of this function.
                double size_of_bound = mesh_bound_multiplier * buffer_constant
                                       * (levelset_function->get_speed() * tau
                                          * bdf_type + h);

                // TODO da jeg la til disse to linjene ble feilen regnet ut riktig
                //  for supplied solution. Betyr dette at dof_handler ikke blir
                //  satt riktig i den metoden?
                setup_level_set();
                cut_mesh_classifier.reclassify();
                dof_handlers.emplace_front(new hp::DoFHandler<dim>());
                distribute_dofs(dof_handlers.front(), size_of_bound);
            }
            
            solutions.emplace_front(locally_owned_dofs, locally_relevant_dofs,
                                    mpi_communicator);

            interpolate_solution(dof_handlers.front(), k);

            // Compute the error for this step.
            errors[k] = compute_error(dof_handlers.front(),
                                      solutions.front());
            errors[k]->time_step = k;
            errors[k]->output();
            this->output_results(dof_handlers.front(),
                                 solutions.front(), k, false);
        }
    }

    template<int dim>
    void CutFEMProblem<dim>::
    set_supplied_solutions(unsigned int bdf_type,
                           std::vector<LA::MPI::Vector> &supplied_solutions,
                           std::vector<std::shared_ptr<hp::DoFHandler<dim>>> &supplied_dof_handlers,
                           std::vector<ErrorBase *> &errors) {
        pcout << "Set supplied solutions" << std::endl;

        // At this point we assume the solution vector that are used for solving
        // the next time step is not yet created.
        assert(solutions.size() == bdf_type);

        if (moving_domain) {
            // If we have a moving domain, then the method
            // interpolate_first_steps should already have created dof_handlers
            // for the interpolated steps, but not yet for the step we wan to
            // solve the equations for next.
            assert(dof_handlers.size() == bdf_type);
        } else {
            // In cases with a stationary domain, a dof_handler is created and
            // accessed with dof_handler.front(), so this is used for all further
            // time steps.
            assert(supplied_dof_handlers.empty());
            // For stationary domains, we only need one dof_handler.
            assert(dof_handlers.size() == 1);
        }

        // Create an extended vector of supplied_solutions, with vectors of
        // length 1 to mark the time steps where we want to keep and use the
        // interpolated solution.
        std::vector<Vector<double>> full_vector(bdf_type, Vector<double>(1));
        std::vector<std::shared_ptr<hp::DoFHandler<dim>>> full_dofs(bdf_type);

        unsigned int num_supp = supplied_solutions.size();
        unsigned int size_diff = bdf_type - num_supp;
        assert(size_diff >= 0);
        for (unsigned int k = 0; k < num_supp; ++k) {
            full_vector[size_diff + k] = supplied_solutions[k];
            if (moving_domain) {
                full_dofs[size_diff + k] = supplied_dof_handlers[k];
            }
        }

        // Insert the supplied solutions in the solutions deque, and compute
        // the errors.
        unsigned int solution_index;
        unsigned int dof_index = 0;

        for (unsigned int k = 0; k < bdf_type; ++k) {
            if (full_vector[k].size() != 1) {
                pcout << " - Set supplied for k = " << k << std::endl;
                // Flip the index, since the solutions deque and the
                // supplied solutions vector holds the solution vectors
                // in opposite order.
                solution_index = solutions.size() - 1 - k;
                // Overwrite the interpolated solution for this step, since it
                // was supplied to the solver.
                solutions[solution_index] = full_vector[k];
                if (moving_domain) {
                    // Replace the previously set dof_handler (of the
                    // interpolated solution), with the supplied one.
                    dof_handlers[solution_index] = full_dofs[k];
                    dof_index = solution_index;
                }

                // Overwrite the error too.
                set_function_times(k * tau);
                // If the domain is stationary, we only have one dof_handler.
                // dof_index = moving_domain ? solution_index : 0;
                errors[k] = compute_error(dof_handlers[dof_index],
                                          solutions[solution_index]);
                errors[k]->time_step = k;
                // TODO feilen i steg u1 er ikke det samme for BDF-2, som den
                //  som blir supplied fra BDF-1.
                errors[k]->output();
            }
        }
    }


    template<int dim>
    void CutFEMProblem<dim>::
    set_function_times(double time) {
        (void) time;
        throw std::logic_error(
                "Override this method to run a time dependent problem.");
    }


    template<int dim>
    void CutFEMProblem<dim>::
    interpolate_solution(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                         int time_step) {
        (void) dof_handler;
        (void) time_step;
        throw std::logic_error(
                "Override this method to run a time dependent problem.");
    }


    template<int dim>
    void CutFEMProblem<dim>::
    set_grid_size() {
        // Save the cell-size, we need it in the Nitsche term and 
        // stabilization parameteres.
        for (auto &cell : triangulation.active_cell_iterators()) {
            if (cell->is_locally_owned()) {
                h = std::pow(cell->measure(), 1.0 / dim);
                break;
            }
        }
    }


    template<int dim>
    void CutFEMProblem<dim>::
    setup_quadrature() {
        const unsigned int n_quad_points = element_order + 1;
        q_collection.push_back(QGauss<dim>(n_quad_points));
        q_collection1D.push_back(QGauss<1>(n_quad_points));
    }


    template<int dim>
    void CutFEMProblem<dim>::
    setup_level_set() {
        pcout << "Setting up level set" << std::endl;
        TimerOutput::Scope t(this->computing_timer, "level set");

        levelset_dof_handler.initialize(triangulation, fe_levelset);
        levelset_dof_handler.distribute_dofs(fe_levelset);

        ls_locally_owned_dofs = levelset_dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(levelset_dof_handler, 
                                                ls_locally_relevant_dofs);

        // The level set function lives on the whole background mesh.
        // Project the geometry onto the mesh.
        LA::MPI::Vector levelset_projection(ls_locally_owned_dofs, mpi_communicator);
        levelset_projection.reinit(ls_locally_owned_dofs, ls_locally_relevant_dofs, mpi_communicator);

        VectorTools::project(mapping_collection[0],
                             levelset_dof_handler,
                             constraints,
                             QGauss<dim>(element_order + 2),
                             *levelset_function,
                             levelset_projection);
        levelset = levelset_projection;
    }


    template<int dim>
    void CutFEMProblem<dim>::
    distribute_dofs(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                    double size_of_bound) {
        // Set outside finite elements to fe, and inside to FE_nothing
        pcout << "Distribute dofs" << std::endl;
        TimerOutput::Scope t(computing_timer, "distribute dofs");
        dof_handler->initialize(triangulation, fe_collection);
        for (const auto &cell : dof_handler->active_cell_iterators()) {
            if (cell->is_locally_owned()) {

                const LocationToLevelSet location =
                        cut_mesh_classifier.location_to_level_set(cell);

                const double distance_from_zero_contour =
                        levelset_function->value(cell->center());

                if (LocationToLevelSet::inside == location ||
                    LocationToLevelSet::intersected == location ||
                    distance_from_zero_contour <= size_of_bound) {
                    // 0 is fe
                    cell->set_active_fe_index(0);
                } else {
                    // 1 is FE_nothing
                    cell->set_active_fe_index(1);
                }
            }
        }
        dof_handler->distribute_dofs(this->fe_collection);
    }


    template<int dim>
    void CutFEMProblem<dim>::
    initialize_stiffness_matrix() {
        pcout << "Initialize marices" << std::endl;
        TimerOutput::Scope t(computing_timer, "initialize matrices");
        
        rhs.reinit(locally_owned_dofs, mpi_communicator);

        DynamicSparsityPattern dsp(locally_relevant_dofs);
        make_sparsity_pattern_for_stabilized(dsp, 
                                             *dof_handlers.front());
        stiffness_matrix.reinit(locally_owned_dofs, 
                                locally_owned_dofs, 
                                dsp, 
                                mpi_communicator);
    }


    template<int dim>
    void CutFEMProblem<dim>::
    initialize_timedep_matrix() {
        pcout << "Initialize timedep marix" << std::endl;
        TimerOutput::Scope t(computing_timer, "initialize timedep matrix");
        
        DynamicSparsityPattern dsp(locally_relevant_dofs);
        make_sparsity_pattern_for_stabilized(dsp, 
                                             *dof_handlers.front());
        // TODO we dont need the same sparsity pattern for 
        // timedep_stiffness_matrix, since it is not stabilized.
        timedep_stiffness_matrix.reinit(locally_owned_dofs,
                                        locally_owned_dofs, 
                                        dsp,
                                        mpi_communicator);
    }


    template<int dim>
    void CutFEMProblem<dim>::
    make_sparsity_pattern_for_stabilized(DynamicSparsityPattern &dsp,
                                         const hp::DoFHandler<dim> &dof_handler) {
        // This method was taken from Simons code, and edited for mpi.
        // TODO put this in utilities.
        // TODO add contraints here
        DoFTools::make_sparsity_pattern(dof_handler, dsp);

        const hp::FECollection<dim> &fe_collection =
            dof_handler.get_fe_collection();
        // Copied from step-46.
        Table<2, DoFTools::Coupling> cell_coupling(fe_collection.n_components(),
                                                    fe_collection.n_components());
        Table<2, DoFTools::Coupling> face_coupling(fe_collection.n_components(),
                                                    fe_collection.n_components());
        for (unsigned int c = 0; c < fe_collection.n_components(); ++c) {
            for (unsigned int d = 0; d < fe_collection.n_components(); ++d) {
                cell_coupling[c][d] = DoFTools::always;
                face_coupling[c][d] = DoFTools::always;
            }
        }
        DoFTools::make_flux_sparsity_pattern(dof_handler,
                                            dsp,
                                            cell_coupling,
                                            face_coupling);

        // constraints.condense(dsp);
        SparsityTools::distribute_sparsity_pattern(
            dsp, dof_handler.locally_owned_dofs(), mpi_communicator,
            locally_relevant_dofs);
    }


    template<int dim>
    void CutFEMProblem<dim>::
    pre_matrix_assembly() {}


    template<int dim>
    void CutFEMProblem<dim>::
    assemble_system() {
        pcout << "Assemble system: matrix and rhs" << std::endl;
        assemble_matrix();
        assemble_rhs(0);
        if (!stationary_stiffness_matrix) {
            assemble_timedep_matrix();
        }
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_local_over_cell(const FEValues<dim> &fe_values,
                             const std::vector<types::global_dof_index> &loc2glb) {
        (void) fe_values;
        (void) loc2glb;
        throw std::logic_error("Not implemented: assemble_local_over_cell");
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_local_over_surface(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {
        (void) fe_values;
        (void) loc2glb;
        throw std::logic_error("Not implemented: asssemble_local_over_surface");
    }


    template<int dim>
    void CutFEMProblem<dim>::
    assemble_matrix() {
        throw std::logic_error("Not implemented: assemble_matrix");
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_timedep_matrix() {
        throw std::logic_error("Not implemented: assemble_timedep_matrix");
    }


    template<int dim>
    void CutFEMProblem<dim>::
    assemble_matrix_local_over_cell(const FEValues<dim> &fe_values,
                                    const std::vector<types::global_dof_index> &loc2glb) {
        (void) fe_values;
        (void) loc2glb;
        throw std::logic_error(
                "Not implemented: assemble_matrix_local_over_cell");
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_matrix_local_over_surface(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {
        (void) fe_values;
        (void) loc2glb;
        throw std::logic_error(
                "Not implemented: assemble_matrix_local_over_surface");
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_rhs(int time_step) {
        (void) time_step;
        throw std::logic_error("Not implemented: assemble_rhs");
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_rhs_local_over_cell(const FEValues<dim> &fe_values,
                                 const std::vector<types::global_dof_index> &loc2glb) {
        (void) fe_values;
        (void) loc2glb;
        throw std::logic_error("Not implemented: assemble_rhs_local_over_cell");
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_rhs_local_over_cell_cn(const FEValues<dim> &fe_values,
                                    const std::vector<types::global_dof_index> &loc2glb,
                                    const int time_step) {
        (void) fe_values;
        (void) loc2glb;
        (void) time_step;
        throw std::logic_error(
                "Not implemented: assemble_rhs_local_over_cell_cn");
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_rhs_local_over_surface(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glob) {
        (void) fe_values;
        (void) loc2glob;
        throw std::logic_error(
                "Not implemented: assemble_rhs_local_over_surface");
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_rhs_local_over_surface_cn(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glob,
            const int time_step) {
        (void) fe_values;
        (void) loc2glob;
        (void) time_step;
        throw std::logic_error(
                "Not implemented: assemble_rhs_local_over_surface_cn");
    }


    template<int dim>
    void CutFEMProblem<dim>::
    solve() {
        pcout << "Solving system" << std::endl;
        TimerOutput::Scope t(computing_timer, "solve");

        if (stationary_stiffness_matrix) {
            SolverControl cn;
            PETScWrappers::SparseDirectMUMPS solver(cn, mpi_communicator);
            solver.set_symmetric_mode(false);
            LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                            mpi_communicator);
            solver.solve(stiffness_matrix, completely_distributed_solution, rhs);
            solutions.front() = completely_distributed_solution;
        } else {
            DynamicSparsityPattern dsp(locally_relevant_dofs);
            make_sparsity_pattern_for_stabilized(dsp, 
                                                 *dof_handlers.front());
            LA::MPI::SparseMatrix timedep;
            timedep.reinit(locally_owned_dofs, 
                           locally_owned_dofs, 
                           dsp, 
                           mpi_communicator);

            timedep.copy_from(stiffness_matrix);
            timedep.add(1, timedep_stiffness_matrix);
            
            SolverControl cn;
            PETScWrappers::SparseDirectMUMPS solver(cn, mpi_communicator);
            solver.set_symmetric_mode(false);
            LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                            mpi_communicator);
            solver.solve(timedep, completely_distributed_solution, rhs);
            solutions.front() = completely_distributed_solution;

        }

        pcout << "   Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;
        pcout << "   Number of degrees of freedom: " << dof_handlers.front()->n_dofs()
              << std::endl;
    }


    template<int dim>
    double CutFEMProblem<dim>::
    compute_condition_number() {
        pcout << "Compute condition number" << std::endl;

        // Invert the stiffness_matrix
        FullMatrix<double> stiffness_matrix_full(solutions.front().size());
        stiffness_matrix_full.copy_from(stiffness_matrix);
        FullMatrix<double> inverse(solutions.front().size());
        inverse.invert(stiffness_matrix_full);

        double norm = stiffness_matrix.frobenius_norm();
        double inverse_norm = inverse.frobenius_norm();

        double condition_number = norm * inverse_norm;
        pcout << "  cond_num = " << condition_number << std::endl;

        // TODO bruk eigenvalues istedet
        return condition_number;
    }
    
    
    template<int dim>
    void CutFEMProblem<dim>::
    output_results(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                   LA::MPI::Vector &solution,
                   int time_step,
                   bool minimal_output) const {
        (void) dof_handler;
        (void) solution;
        (void) time_step;
        (void) minimal_output;
    }

    template<int dim>
    void CutFEMProblem<dim>::
    output_results(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                   LA::MPI::Vector &solution,
                   std::string &suffix,
                   bool minimal_output) const {
        (void) dof_handler;
        (void) solution;
        (void) suffix;
        (void) minimal_output;
        // TODO remove this unused old method.
    }

    template<int dim>
    void CutFEMProblem<dim>::
    output_results(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                   LA::MPI::Vector &solution,
                   bool minimal_output) const {
        output_results(dof_handler, solution, 0, minimal_output);
    }

    template<int dim>
    void CutFEMProblem<dim>::
    output_levelset(int time_step) const {
        DataOut<dim> data_out_levelset;
        data_out_levelset.attach_dof_handler(levelset_dof_handler);
        data_out_levelset.add_data_vector(levelset, "levelset");
        data_out_levelset.build_patches();

        // Write the subdomains.
        Vector<float> subdomain(triangulation.n_active_cells());
        for (unsigned int i = 0; i < subdomain.size(); ++i) {
            subdomain(i) = triangulation.locally_owned_subdomain();
        }
        data_out_levelset.add_data_vector(subdomain, "subdomain");
        data_out_levelset.build_patches();

        data_out_levelset.write_vtu_with_pvtu_record(
            "", "levelset-d" + std::to_string(dim)
                + "o" + std::to_string(element_order)
                + "r" + std::to_string(n_refines),
            time_step, mpi_communicator, 2, 8);
    }


    template<int dim>
    void CutFEMProblem<dim>::
    post_processing(unsigned int time_step) {
        (void) time_step;
    }

    template
    class LevelSet<2>;

    template
    class LevelSet<3>;

    template
    class CutFEMProblem<2>;

    template
    class CutFEMProblem<3>;

}

