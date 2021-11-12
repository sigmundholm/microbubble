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
        dof_handlers.emplace_front(new hp::DoFHandler<dim>(triangulation));
        setup_fe_collection();
        distribute_dofs(dof_handlers.front());
        initialize_matrices();
        int n_dofs = dof_handlers.front()->n_dofs();
        solutions.emplace_front(n_dofs);
        pre_matrix_assembly();
        this->assemble_system();
        solve();
        if (write_output) {
            output_results(this->dof_handlers.front(),
                           this->solutions.front());
        }
        return compute_error(dof_handlers.front(), solutions.front());
    }


    template<int dim>
    Vector<double> CutFEMProblem<dim>::
    get_solution() {
        return solutions.front();
    }


    template<int dim>
    std::shared_ptr<hp::DoFHandler<dim>> CutFEMProblem<dim>::
    get_dof_handler() {
        // TODO return reference or pointer?
        return dof_handlers.front();
    }


    template<int dim>
    ErrorBase *CutFEMProblem<dim>::
    run_time(unsigned int bdf_type, unsigned int steps,
             std::vector<Vector<double>> &supplied_solutions) {

        std::cout << "\nBDF-" << bdf_type << ", steps=" << steps << std::endl;
        std::cout << "-------------------------" << std::endl;
        // TODO fix BDF-2 with given u1 from BDF-1

        make_grid(triangulation);
        setup_quadrature();
        setup_level_set();
        cut_mesh_classifier.reclassify();
        setup_fe_collection();
        // Initialize the first dof_handler.
        dof_handlers.emplace_front(new hp::DoFHandler<dim>());
        distribute_dofs(dof_handlers.front());
        initialize_matrices();

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
            int n_dofs = this->dof_handlers.front()->n_dofs();
            this->solutions.emplace_front(n_dofs);

            if (k == bdf_type) {
                // Assemble the matrix after the new solution vector is created.
                // This is to omit index problems when assembling a stiffness
                // matrix that is dependent on previous solutions.
                pre_matrix_assembly();
                assemble_matrix();
            }

            // TODO nødvendig??
            this->rhs.reinit(this->solutions.front().size());

            assemble_rhs(k);
            this->solve();
            errors[k] = this->compute_error(dof_handlers.front(),
                                            solutions.front());
            errors[k]->time_step = k;
            write_time_error_to_file(errors[k], file);
            errors[k]->output();

            if (this->write_output) {
                this->output_results(this->dof_handlers.front(),
                                     this->solutions.front(), k, true);
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
                      std::vector<std::shared_ptr<hp::DoFHandler<dim>>> &supplied_dof_handlers,
                      const double mesh_bound_multiplier) {

        std::cout << "\nBDF-" << bdf_type << ", steps=" << steps << std::endl;
        std::cout << "-------------------------" << std::endl;
        moving_domain = true;

        // One dof_handler must be supplied for each supplied solution vector.
        assert(supplied_solutions.size() == supplied_dof_handlers.size());
        solutions.clear();
        dof_handlers.clear();

        if (triangulation.n_quads() == 0) {
            make_grid(triangulation);
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
        std::cout << " # size_of_bound = " << size_of_bound << std::endl;

        dof_handlers.emplace_front(new hp::DoFHandler<dim>());
        distribute_dofs(dof_handlers.front(), size_of_bound);

        initialize_matrices();

        // Vector for the computed error for each time step.
        std::vector<ErrorBase *> errors(steps + 1);

        interpolate_first_steps(bdf_type, errors, mesh_bound_multiplier);
        set_supplied_solutions(bdf_type, supplied_solutions,
                               supplied_dof_handlers, errors, true);
        set_bdf_coefficients(bdf_type);
        set_extrapolation_coefficients(bdf_type);

        std::ofstream file("errors-time-d" + std::to_string(dim)
                           + "o" + std::to_string(element_order)
                           + "r" + std::to_string(n_refines) + ".csv");
        write_time_header_to_file(file);

        std::cout << "Interpolated / supplied solutions." << std::endl;
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
            int n_dofs = dof_handlers.front()->n_dofs();
            solutions.emplace_front(n_dofs);

            // Redistribute the dofs after the level set was updated
            // size_of_bound = buffer_constant * bdf_type * this->h;
            size_of_bound = mesh_bound_multiplier * buffer_constant
                            * (levelset_function->get_speed() * tau
                               * bdf_type + h);
            std::cout << " # size_of_bound = " << size_of_bound << std::endl;
            dof_handlers.emplace_front(new hp::DoFHandler<dim>());
            distribute_dofs(dof_handlers.front(), size_of_bound);

            // Reinitialize the matrices and vectors after the number of dofs
            // was updated.
            initialize_matrices();

            pre_matrix_assembly();
            assemble_matrix();
            assemble_rhs(k);

            solve();
            errors[k] = compute_error(dof_handlers.front(), solutions.front());
            errors[k]->time_step = k;
            write_time_error_to_file(errors[k], file);
            errors[k]->output();

            if (write_output) {
                output_results(this->dof_handlers.front(),
                               this->solutions.front(), k, false);
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
    run_moving_domain(unsigned int bdf_type, unsigned int steps,
                      const double mesh_bound_multiplier) {
        // Invoking this method will result in a pure BDF-k method, where all
        // the initial steps will be interpolated.
        std::vector<Vector<double>> empty_solutions;
        std::vector<std::shared_ptr<hp::DoFHandler<dim>>> empty_dof_h;
        return run_moving_domain(bdf_type, steps, empty_solutions, empty_dof_h,
                                 mesh_bound_multiplier);
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
        std::cout << "Interpolate first step(s)." << std::endl;
        // Assume the deque of solutions is empty at this point.
        assert(solutions.empty());
        // At this point, one dof_handler should have been created.
        assert(dof_handlers.size() == 1);

        for (unsigned int k = 0; k < bdf_type; ++k) {
            // Create a new solution vector.
            std::cout << " - Interpolate step k = " << k << std::endl;

            // Interpolate it a the correct time.
            set_function_times(k * this->tau);

            if (moving_domain && k > 0) {
                // For moving domains we need a new dof_handler for each step,
                // but the first one should already have been created.
                double buffer_constant = 2;
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
            int n_dofs = dof_handlers.front()->n_dofs();
            solutions.emplace_front(n_dofs);

            interpolate_solution(dof_handlers.front(), k);

            // Compute the error for this step.
            errors[k] = this->compute_error(dof_handlers.front(),
                                            solutions.front());
            errors[k]->time_step = k;
            errors[k]->output();
            std::string suffix = std::to_string(k) + "-inter";
            this->output_results(this->dof_handlers.front(),
                                 this->solutions.front(), suffix);
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
                           std::vector<std::shared_ptr<hp::DoFHandler<dim>>> &supplied_dof_handlers,
                           std::vector<ErrorBase *> &errors,
                           bool moving_domain) {
        std::cout << "Set supplied solutions" << std::endl;

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
        std::vector<Vector<double>>
                full_vector(bdf_type, Vector<double>(1));
        std::vector<std::shared_ptr<hp::DoFHandler<dim>>> full_dofs(
                bdf_type); //, nullptr);
        // TODO bruk shared_ptr istedet for reference_wrap
        unsigned int num_supp = supplied_solutions.size();
        unsigned int size_diff = bdf_type - num_supp;
        assert(size_diff >= 0);
        for (unsigned int k = 0; k < num_supp; ++k) {
            full_vector[size_diff + k] = supplied_solutions[k];
            full_dofs[size_diff + k] = supplied_dof_handlers[k];
        }

        // Insert the supplied solutions in the solutions deque, and compute
        // the errors.
        unsigned int solution_index;
        unsigned int dof_index = 0;
        unsigned int n_dofs = this->dof_handlers.front()->n_dofs();

        for (unsigned int k = 0; k < bdf_type; ++k) {
            if (full_vector[k].size() != 1) {
                std::cout << " - Set supplied for k = " << k << std::endl;
                // Flip the index, since the solutions deque and the
                // supplied solutions vector holds the solution vectors
                // in opposite order.
                solution_index = solutions.size() - 1 - k;
                // Overwrite the interpolated solution for this step, since it
                // was supplied to the solver.
                solutions[solution_index] = full_vector[k];
                if (moving_domain) {
                    // TODO ignore the dof_handlers if the vector is empty,
                    //  then the run_time method was run.
                    // Replace the previously set dof_handler (of the
                    // interpolated solution), with the supplied one.
                    dof_handlers[solution_index] = full_dofs[k];
                    dof_index = solution_index;
                }

                // Overwrite the error too.
                set_function_times(k * this->tau);
                // If the domain is stationary, we only have one dof_handler.
                // dof_index = moving_domain ? solution_index : 0;
                errors[k] = this->compute_error(dof_handlers[dof_index],
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
        throw std::logic_error(
                "Override this method to run a time dependent problem.");
    }


    template<int dim>
    void CutFEMProblem<dim>::
    interpolate_solution(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                         int time_step) {
        throw std::logic_error(
                "Override this method to run a time dependent problem.");
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
    distribute_dofs(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                    double size_of_bound) {
        // Set outside finite elements to fe, and inside to FE_nothing
        dof_handler->initialize(triangulation, fe_collection);
        for (const auto &cell : dof_handler->active_cell_iterators()) {
            const LocationToLevelSet location =
                    cut_mesh_classifier.location_to_level_set(cell);

            const double distance_from_zero_contour =
                    levelset_function->value(
                            cell->center());

            if (LocationToLevelSet::INSIDE == location ||
                LocationToLevelSet::INTERSECTED == location ||
                distance_from_zero_contour <= size_of_bound) {
                // 0 is fe
                cell->set_active_fe_index(0);
            } else {
                // 1 is FE_nothing
                cell->set_active_fe_index(1);
            }
        }
        dof_handler->distribute_dofs(this->fe_collection);
    }


    template<int dim>
    void CutFEMProblem<dim>::
    initialize_matrices() {
        std::cout << "Initialize marices" << std::endl;
        int n_dofs = dof_handlers.front()->n_dofs();
        rhs.reinit(n_dofs);

        // TODO unopack the pointer in some way?
        cutfem::nla::make_sparsity_pattern_for_stabilized(*dof_handlers.front(),
                                                          sparsity_pattern);
        stiffness_matrix.reinit(sparsity_pattern);
    }


    template<int dim>
    void CutFEMProblem<dim>::
    pre_matrix_assembly() {}


    template<int dim>
    void CutFEMProblem<dim>::
    assemble_system() {
        throw std::logic_error("Not implemented.");
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_local_over_cell(const FEValues<dim> &fe_values,
                             const std::vector<types::global_dof_index> &loc2glb) {
        throw std::logic_error("Not implemented.");
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_local_over_surface(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {
        throw std::logic_error("Not implemented.");
    }


    template<int dim>
    void CutFEMProblem<dim>::
    assemble_matrix() {
        throw std::logic_error("Not implemented.");
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_matrix_local_over_cell(const FEValues<dim> &fe_values,
                                    const std::vector<types::global_dof_index> &loc2glb) {
        throw std::logic_error("Not implemented.");
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_matrix_local_over_surface(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {
        throw std::logic_error("Not implemented.");
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_rhs(int time_step) {
        throw std::logic_error("Not implemented.");
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_rhs_local_over_cell(const FEValues<dim> &fe_values,
                                 const std::vector<types::global_dof_index> &loc2glb) {
        throw std::logic_error("Not implemented.");
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_rhs_local_over_cell_cn(const FEValues<dim> &fe_values,
                                    const std::vector<types::global_dof_index> &loc2glb,
                                    const int time_step) {
        throw std::logic_error("Not implemented.");
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_rhs_local_over_surface(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glob) {
        throw std::logic_error("Not implemented.");
    }

    template<int dim>
    void CutFEMProblem<dim>::
    assemble_rhs_local_over_surface_cn(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glob,
            const int time_step) {
        throw std::logic_error("Not implemented.");
    }


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
    output_results(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                   Vector<double> &solution,
                   bool minimal_output) const {
        std::string empty;
        output_results(dof_handler, solution, empty, minimal_output);
    }


    template<int dim>
    void CutFEMProblem<dim>::
    output_results(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                   Vector<double> &solution,
                   int time_step, bool minimal_output) const {
        std::string k = std::to_string(time_step);
        output_results(dof_handler, solution, k, minimal_output);
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

