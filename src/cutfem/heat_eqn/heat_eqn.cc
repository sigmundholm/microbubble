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

#include <deal.II/non_matching/cut_mesh_classifier.h>
#include <deal.II/non_matching/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/optional.hpp>

#include <assert.h>
#include <cmath>
#include <fstream>

#include "cutfem/nla/sparsity_pattern.h"
#include "cutfem/stabilization/jump_stabilization.h"

#include "heat_eqn.h"


using namespace cutfem;


namespace examples::cut::HeatEquation {


    template<int dim>
    HeatEqn<dim>::HeatEqn(const double nu,
                          const double tau,
                          const double radius,
                          const double half_length,
                          const unsigned int n_refines,
                          const int element_order,
                          const bool write_output,
                          Function<dim> &rhs,
                          Function<dim> &bdd_values,
                          Function<dim> &analytical_soln,
                          Function<dim> &levelset_func,
                          const bool stabilized,
                          const bool crank_nicholson)
            : ScalarProblem<dim>(n_refines, element_order, write_output,
                                 levelset_func, analytical_soln, stabilized),
              nu(nu), tau(tau), radius(radius), half_length(half_length),
              crank_nicholson(crank_nicholson) {
        // Use no constraints when projecting.
        this->constraints.close();

        this->rhs_function = &rhs;
        this->boundary_values = &bdd_values;
    }


    template<int dim>
    Error HeatEqn<dim>::
    run(unsigned int bdf_type, unsigned int steps,
        Vector<double> &supplied_solution) {
        // TODO imlement bdf2
        if (crank_nicholson) {
            std::cout << "\nCrank-Nicholson" << std::endl;
        } else {
            std::cout << "\nBDF-" << bdf_type << std::endl;
        }

        if (!triangulation_exists) {
            make_grid(this->triangulation);
            this->setup_quadrature();
            this->setup_level_set();
            this->cut_mesh_classifier.reclassify();
            this->distribute_dofs();
            this->initialize_matrices();
        }

        if (crank_nicholson) {
            assert(bdf_type == 1);
        }

        std::vector<Error> errors(steps + 1);
        interpolate_first_steps(bdf_type, errors);
        set_bdf_coefficients(bdf_type);

        assemble_matrix();

        // TODO BDF-2: if u1 is provided; compute the error that step.
        std::ofstream file("errors-time-d" + std::to_string(dim)
                           + "o" + std::to_string(this->element_order)
                           + "r" + std::to_string(this->n_refines) + ".csv");
        write_time_header_to_file(file);

        // Overwrite the interpolated solution if the supplied_solution is a
        // vector of lenght longer than one.
        if (supplied_solution.size() == this->solution.size()) {
            std::cout << "BDF-" << bdf_type << ", supplied solution set."
                      << std::endl;
            solutions[bdf_type - 1] = supplied_solution;
            this->solution = supplied_solution;
            this->analytical_solution->set_time((bdf_type - 1) * tau);
            errors[bdf_type - 1] = this->compute_error();
            errors[bdf_type - 1].time_step = bdf_type - 1;
        }

        // Write the interpolation errors to file.
        // TODO note that this results in both the interpolation error and the
        //  fem error to be written when u1 is supplied to bdf-2.
        for (unsigned int k = 0; k < bdf_type; ++k) {
            write_time_error_to_file(errors[k], file);
            std::cout << "  k = " << k << ", "
                      << "|| u - u_h ||_L2 = " << errors[k].l2_error
                      << ", || u - u_h ||_H1 = " << errors[k].h1_error
                      << std::endl;
        }

        for (unsigned int k = bdf_type; k <= steps; ++k) {
            std::cout << "\nk = " << std::to_string(k)
                      << ", time = " << std::to_string(k * tau)
                      << ", tau = " << std::to_string(tau) << std::endl;
            std::cout << "-------------------------" << std::endl;

            this->rhs_function->set_time(k * tau);
            this->boundary_values->set_time(k * tau);
            this->analytical_solution->set_time(k * tau);

            assemble_rhs(k);
            this->solve();
            errors[k] = this->compute_error();
            errors[k].time_step = k;
            write_time_error_to_file(errors[k], file);

            std::cout << "  k = " << k << ", "
                      << "|| u - u_h ||_L2 = " << errors[k].l2_error
                      << ", || u - u_h ||_H1 = " << errors[k].h1_error
                      << std::endl;

            std::string suffix = "-" + std::to_string(k);
            if (this->write_output) {
                this->output_results(suffix, false);
            }

            for (unsigned long i = 1; i < solutions.size(); ++i) {
                solutions[i - 1] = solutions[i];
            }
            solutions[solutions.size() - 1] = this->solution;
        }

        // compute_condition_number();
        return compute_time_error(errors);
    }


    template<int dim>
    Error HeatEqn<dim>::
    run(unsigned int bdf_type, unsigned int steps) {
        Vector<double> empty(1);
        return run(bdf_type, steps, empty);
    }


    template<int dim>
    void HeatEqn<dim>::
    set_bdf_coefficients(unsigned int bdf_type) {
        bdf_coeffs = std::vector<double>(bdf_type + 1);

        if (bdf_type == 1) {
            bdf_coeffs[0] = -1;
            bdf_coeffs[1] = 1;
        } else if (bdf_type == 2) {
            bdf_coeffs[0] = 0.5;
            bdf_coeffs[1] = -2;
            bdf_coeffs[2] = 1.5;
        } else if (bdf_type == 3) {
            bdf_coeffs[0] = -1.0 / 3;
            bdf_coeffs[1] = 1.5;
            bdf_coeffs[2] = -3;
            bdf_coeffs[3] = 11.0 / 6;
        } else {
            throw std::invalid_argument("Only BDF-1 is implemented for now.");
        }
    }


    template<int dim>
    void HeatEqn<dim>::
    interpolate_first_steps(unsigned int bdf_type, std::vector<Error> &errors) {
        solutions = std::vector<Vector<double>>(bdf_type);

        std::cout << "Interpolate first step(s)" << std::endl;

        for (unsigned int i = 0; i < bdf_type; ++i) {
            // Interpolate step i (step u1 will be overwritten by bdf2 if
            // u1 is provided).
            std::cout << "  Interpolate step k = " << i
                      << ", time = " << i * tau << std::endl;
            this->analytical_solution->set_time(i * tau);
            VectorTools::interpolate(this->dof_handler,
                                     *(this->analytical_solution),
                                     this->solution);
            solutions[i].reinit(this->solution.size());

            // Compute the error of the interpolated step.
            errors[i] = this->compute_error();
            errors[i].time_step = i;
            std::string suffix = "-" + std::to_string(i) + "-inter";
            this->output_results(suffix);
            solutions[i] = this->solution;
        }
    }


    template<int dim>
    void
    HeatEqn<dim>::make_grid(Triangulation<dim> &tria) {
        std::cout << "Creating triangulation" << std::endl;

        triangulation_exists = true;
        GridGenerator::cylinder(tria, radius, half_length);
        GridTools::remove_anisotropy(tria, 1.618, 5);
        tria.refine_global(this->n_refines);

        this->mapping_collection.push_back(MappingCartesian<dim>());

        // Save the cell-size, we need it in the Nitsche term.
        typename Triangulation<dim>::active_cell_iterator cell =
                tria.begin_active();
        this->h = std::pow(cell->measure(), 1.0 / dim);
    }


    template<int dim>
    void HeatEqn<dim>::
    assemble_local_over_cell(const FEValues<dim> &fe_values,
                             const std::vector<types::global_dof_index> &loc2glb) {
        throw std::logic_error("One step run not available for Heat Equation");
    }


    template<int dim>
    void HeatEqn<dim>::
    assemble_local_over_surface(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {
        throw std::logic_error("One step run not available for Heat Equation");
    }

    template<int dim>
    void HeatEqn<dim>::
    assemble_matrix() {
        std::cout << "Assembling" << std::endl;

        this->stiffness_matrix = 0;
        this->rhs = 0;

        // Use a helper object to compute the stabilisation for both the velocity
        // and the pressure component.
        const FEValuesExtractors::Scalar velocities(0);
        stabilization::JumpStabilization<dim, FEValuesExtractors::Scalar>
                velocity_stabilization(this->dof_handler,
                                       this->mapping_collection,
                                       this->cut_mesh_classifier,
                                       this->constraints);
        if (this->stabilized) {
            velocity_stabilization.set_function_describing_faces_to_stabilize(
                    stabilization::inside_stabilization);
            velocity_stabilization.set_weight_function(
                    stabilization::taylor_weights);
            velocity_stabilization.set_extractor(velocities);
        }

        NonMatching::RegionUpdateFlags region_update_flags;
        region_update_flags.inside = update_values | update_JxW_values |
                                     update_gradients |
                                     update_quadrature_points;
        region_update_flags.surface = update_values | update_JxW_values |
                                      update_gradients |
                                      update_quadrature_points |
                                      update_normal_vectors;

        NonMatching::FEValues<dim> cut_fe_values(this->mapping_collection,
                                                 this->fe_collection,
                                                 this->q_collection,
                                                 this->q_collection1D,
                                                 region_update_flags,
                                                 this->cut_mesh_classifier,
                                                 this->levelset_dof_handler,
                                                 this->levelset);

        // Quadrature for the faces of the cells on the outer boundary
        QGauss<dim - 1> face_quadrature_formula(this->fe.degree + 1);
        FEFaceValues<dim> fe_face_values(this->fe,
                                         face_quadrature_formula,
                                         update_values | update_gradients |
                                         update_quadrature_points |
                                         update_normal_vectors |
                                         update_JxW_values);

        double beta_0 = 0.1;
        double gamma_A =
                beta_0 * this->element_order * (this->element_order + 1);
        double gamma_M =
                beta_0 * this->element_order * (this->element_order + 1);

        for (const auto &cell : this->dof_handler.active_cell_iterators()) {
            const unsigned int n_dofs = cell->get_fe().dofs_per_cell;
            std::vector<types::global_dof_index> loc2glb(n_dofs);
            cell->get_dof_indices(loc2glb);

            // This call will compute quadrature rules relevant for this cell
            // in the background.
            cut_fe_values.reinit(cell);

            // Retrieve an FEValues object with quadrature points
            // over the full cell.
            const boost::optional<const FEValues<dim> &> fe_values_bulk =
                    cut_fe_values.get_inside_fe_values();

            if (fe_values_bulk) {
                assemble_matrix_local_over_cell(*fe_values_bulk, loc2glb);
            }

            // Retrieve an FEValues object with quadrature points
            // on the immersed surface.
            const boost::optional<const FEImmersedSurfaceValues<dim> &>
                    fe_values_surface = cut_fe_values.get_surface_fe_values();

            if (fe_values_surface)
                assemble_matrix_local_over_surface(*fe_values_surface, loc2glb);

            if (this->stabilized) {
                // Compute and add the velocity stabilization.
                velocity_stabilization.compute_stabilization(cell);
                velocity_stabilization.add_stabilization_to_matrix(
                        tau * gamma_M +
                        tau * nu * gamma_A / (this->h * this->h),
                        this->stiffness_matrix);
            }
        }
    }

    template<int dim>
    void
    HeatEqn<dim>::assemble_matrix_local_over_cell(
            const FEValues<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {
        // TODO generelt: er det for mange hjelpeobjekter som opprettes her i cella?
        //  bør det heller gjøres i funksjonen før og sendes som argumenter? hvis
        //  det er mulig mtp cellene som blir cut da

        // Matrix and vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // Calculate often used terms in the beginning of each cell-loop
        std::vector<double> phi(dofs_per_cell);
        std::vector<Tensor<1, dim>> grad_phi(dofs_per_cell);

        double cn_factor = crank_nicholson ? 0.5 : 1;

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
            for (const unsigned int k : fe_values.dof_indices()) {
                grad_phi[k] = fe_values.shape_grad(k, q);
                phi[k] = fe_values.shape_value(k, q);
            }

            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    local_matrix(i, j) += (bdf_coeffs[solutions.size()]
                                           * phi[j] * phi[i]
                                           +
                                           cn_factor * tau * nu * grad_phi[j] *
                                           grad_phi[i]
                                          ) * fe_values.JxW(q); // dx
                }
            }
        }
        this->stiffness_matrix.add(loc2glb, local_matrix);
    }


    template<int dim>
    void
    HeatEqn<dim>::assemble_matrix_local_over_surface(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {
        // Matrix and vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // Calculate often used terms in the beginning of each cell-loop
        std::vector<Tensor<1, dim>> grad_phi(dofs_per_cell);
        std::vector<double> phi(dofs_per_cell);

        double gamma = 20 * this->element_order * (this->element_order + 1);
        double mu = gamma / this->h;
        Tensor<1, dim> normal;
        double cn_factor = crank_nicholson ? 0.5 : 1;

        for (unsigned int q : fe_values.quadrature_point_indices()) {
            normal = fe_values.normal_vector(q);

            for (const unsigned int k : fe_values.dof_indices()) {
                phi[k] = fe_values.shape_value(k, q);
                grad_phi[k] = fe_values.shape_grad(k, q);
            }

            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    local_matrix(i, j) +=
                            cn_factor * tau * nu * (
                                    mu * phi[j] * phi[i]  // mu (u, v)
                                    -
                                    grad_phi[j] * normal * phi[i] // (∂_n u,v)
                                    -
                                    phi[j] * grad_phi[i] * normal // (u,∂_n v)
                            ) * fe_values.JxW(q); // ds
                }
            }
        }
        this->stiffness_matrix.add(loc2glb, local_matrix);
    }

    template<int dim>
    void
    HeatEqn<dim>::assemble_rhs(int time_step) {
        std::cout << "Assembling RHS" << std::endl;

        this->rhs = 0;

        NonMatching::RegionUpdateFlags region_update_flags;
        region_update_flags.inside = update_values | update_JxW_values |
                                     update_gradients |
                                     update_quadrature_points;
        region_update_flags.surface = update_values | update_JxW_values |
                                      update_gradients |
                                      update_quadrature_points |
                                      update_normal_vectors;

        NonMatching::FEValues<dim> cut_fe_values(this->mapping_collection,
                                                 this->fe_collection,
                                                 this->q_collection,
                                                 this->q_collection1D,
                                                 region_update_flags,
                                                 this->cut_mesh_classifier,
                                                 this->levelset_dof_handler,
                                                 this->levelset);

        // Quadrature for the faces of the cells on the outer boundary
        QGauss<dim - 1> face_quadrature_formula(this->fe.degree + 1);
        FEFaceValues<dim> fe_face_values(this->fe,
                                         face_quadrature_formula,
                                         update_values | update_gradients |
                                         update_quadrature_points |
                                         update_normal_vectors |
                                         update_JxW_values);

        for (const auto &cell : this->dof_handler.active_cell_iterators()) {
            const unsigned int n_dofs = cell->get_fe().dofs_per_cell;
            std::vector<types::global_dof_index> loc2glb(n_dofs);
            cell->get_dof_indices(loc2glb);

            // This call will compute quadrature rules relevant for this cell
            // in the background.
            cut_fe_values.reinit(cell);

            // Retrieve an FEValues object with quadrature points
            // over the full cell.
            const boost::optional<const FEValues<dim> &> fe_values_bulk =
                    cut_fe_values.get_inside_fe_values();

            if (fe_values_bulk) {
                if (crank_nicholson) {
                    assemble_rhs_local_over_cell_cn(*fe_values_bulk, loc2glb,
                                                    time_step);
                } else {
                    assemble_rhs_local_over_cell(*fe_values_bulk, loc2glb);
                }
            }

            // Retrieve an FEValues object with quadrature points
            // on the immersed surface.
            const boost::optional<const FEImmersedSurfaceValues<dim> &>
                    fe_values_surface = cut_fe_values.get_surface_fe_values();

            if (fe_values_surface) {
                if (crank_nicholson) {
                    assemble_rhs_local_over_surface_cn(*fe_values_surface,
                                                       loc2glb, time_step);
                } else {
                    assemble_rhs_local_over_surface(*fe_values_surface,
                                                    loc2glb);
                }

            }
        }
    }

    template<int dim>
    void
    HeatEqn<dim>::assemble_rhs_local_over_cell(
            const FEValues<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {
        // TODO generelt: er det for mange hjelpeobjekter som opprettes her i cella?
        //  bør det heller gjøres i funksjonen før og sendes som argumenter? hvis
        //  det er mulig mtp cellene som blir cut da

        // Vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
        Vector<double> local_rhs(dofs_per_cell);

        // Vector for values of the RightHandSide for all quadrature points on a cell.
        std::vector<double> rhs_values(fe_values.n_quadrature_points);
        this->rhs_function->value_list(fe_values.get_quadrature_points(),
                                       rhs_values);

        // Create vector of the previous solutions values
        std::vector<double> val(fe_values.n_quadrature_points, 0);
        std::vector<std::vector<double>> prev_solution_values(solutions.size(),
                                                              val);

        // The the values of the previous solutions, and insert into the
        // matrix initialized above.
        for (unsigned long k = 0; k < solutions.size(); ++k) {
            fe_values.get_function_values(solutions[k],
                                          prev_solution_values[k]);
        }
        double phi_iq;
        double prev_values;
        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
            for (const unsigned int i : fe_values.dof_indices()) {

                prev_values = 0;
                for (unsigned long k = 0; k < solutions.size(); ++k) {
                    prev_values += bdf_coeffs[k] * prev_solution_values[k][q];
                }

                phi_iq = fe_values.shape_value(i, q);
                local_rhs(i) += (tau * rhs_values[q] * phi_iq // (f, v)
                                 - prev_values * phi_iq       // (u_n, v)
                                ) * fe_values.JxW(q);         // dx
            }
        }
        this->rhs.add(loc2glb, local_rhs);
    }


    /**
     * This calculates the RHS terms that are needed when using
     * Crank-Nicholson as the time stepping method.
     *
     * @tparam dim
     * @param fe_values
     * @param loc2glb
     * @param time_step
     */
    template<int dim>
    void
    HeatEqn<dim>::assemble_rhs_local_over_cell_cn(
            const FEValues<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb,
            const int time_step) {
        // Crank-Nicholson can only be used when a one step method is run.
        assert(solutions.size() == 1 && bdf_coeffs.size() == 2);

        // std::cout << "  rhs cell k = " << time_step << std::endl;

        // Matrix and vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
        Vector<double> local_rhs(dofs_per_cell);

        // Vector for values of the RightHandSide for all quadrature points on a cell.
        std::vector<double> rhs_values(fe_values.n_quadrature_points);
        this->rhs_function->set_time(time_step * tau);
        this->rhs_function->value_list(fe_values.get_quadrature_points(),
                                       rhs_values);

        // Compute the rhs values from the previous time step.
        this->rhs_function->set_time((time_step - 1) * tau);
        std::vector<double> rhs_values_prev(fe_values.n_quadrature_points);
        this->rhs_function->value_list(fe_values.get_quadrature_points(),
                                       rhs_values_prev);

        // Get the previous solution values.
        std::vector<double> prev_solution_values(fe_values.n_quadrature_points,
                                                 0);
        fe_values.get_function_values(this->solution, prev_solution_values);

        // Get the previous solution gradients.
        std::vector<Tensor<1, dim>> prev_solution_grads(
                fe_values.n_quadrature_points, Tensor<1, dim>());
        fe_values.get_function_gradients(this->solution, prev_solution_grads);

        double phi;
        Tensor<1, dim> grad_phi;
        double prev_value;
        Tensor<1, dim> prev_grad;
        double rhs_values_sum;

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
            for (const unsigned int i : fe_values.dof_indices()) {

                phi = fe_values.shape_value(i, q);
                grad_phi = fe_values.shape_grad(i, q);
                prev_value = prev_solution_values[q];
                prev_grad = prev_solution_grads[q];
                rhs_values_sum = rhs_values[q] + rhs_values_prev[q];


                local_rhs(i) += (0.5 * tau * (
                        rhs_values_sum * phi          // (f_n+1 + f_n, v)
                        - nu * prev_grad * grad_phi)  // -ν(∇u_n, ∇v)
                                 + prev_value * phi   // (u_n, v)
                                ) * fe_values.JxW(q); // dx
            }
        }
        this->rhs.add(loc2glb, local_rhs);
    }


    template<int dim>
    void
    HeatEqn<dim>::assemble_rhs_local_over_surface(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {
        // Matrix and vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // Evaluate the boundary function for all quadrature points on this face.
        std::vector<double> bdd_values(fe_values.n_quadrature_points);
        this->boundary_values->value_list(fe_values.get_quadrature_points(),
                                          bdd_values);

        // Calculate often used terms in the beginning of each cell-loop
        std::vector<Tensor<1, dim>> grad_phi(dofs_per_cell);
        std::vector<double> phi(dofs_per_cell);

        double gamma = 20 * this->element_order * (this->element_order + 1);
        double mu = gamma / this->h;
        Tensor<1, dim> normal;

        for (unsigned int q : fe_values.quadrature_point_indices()) {
            normal = fe_values.normal_vector(q);
            for (const unsigned int i : fe_values.dof_indices()) {
                phi[i] = fe_values.shape_value(i, q);
                grad_phi[i] = fe_values.shape_grad(i, q);

                local_rhs(i) +=
                        tau * nu * (mu * bdd_values[q] * phi[i] // mu (g, v)
                                    -
                                    bdd_values[q] * grad_phi[i] *
                                    normal // (g, n ∂_n v)
                        ) * fe_values.JxW(q);        // ds
            }
        }
        this->rhs.add(loc2glb, local_rhs);
    }


    template<int dim>
    void
    HeatEqn<dim>::assemble_rhs_local_over_surface_cn(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb,
            const int time_step) {
        // Matrix and vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // std::cout << "  rhs surf k = " << time_step << std::endl;

        // Evaluate the boundary function for all quadrature points on this face.
        std::vector<double> bdd_values(fe_values.n_quadrature_points);
        this->boundary_values->set_time(time_step * tau);
        this->boundary_values->value_list(fe_values.get_quadrature_points(),
                                          bdd_values);

        std::vector<double> bdd_values_prev(fe_values.n_quadrature_points);
        this->boundary_values->set_time((time_step - 1) * tau);
        this->boundary_values->value_list(fe_values.get_quadrature_points(),
                                          bdd_values_prev);

        // Get the previous solution values.
        std::vector<double> prev_solution_values(fe_values.n_quadrature_points,
                                                 0);
        fe_values.get_function_values(this->solution, prev_solution_values);

        // Get the previous solution gradients.
        std::vector<Tensor<1, dim>> prev_solution_grads(
                fe_values.n_quadrature_points, Tensor<1, dim>());
        fe_values.get_function_gradients(this->solution, prev_solution_grads);

        double gamma = 20 * this->element_order * (this->element_order + 1);
        double mu = gamma / this->h;

        Tensor<1, dim> normal;
        double bdd_values_sum;
        double phi;
        Tensor<1, dim> grad_phi;
        double prev_value;
        Tensor<1, dim> prev_grad;

        for (unsigned int q : fe_values.quadrature_point_indices()) {
            normal = fe_values.normal_vector(q);
            for (const unsigned int i : fe_values.dof_indices()) {
                phi = fe_values.shape_value(i, q);
                grad_phi = fe_values.shape_grad(i, q);
                bdd_values_sum = bdd_values[q] + bdd_values_prev[q];
                prev_value = prev_solution_values[q];
                prev_grad = prev_solution_grads[q];

                local_rhs(i) += 0.5 * tau * nu * (
                        mu * bdd_values_sum * phi // mu (g, v)
                        -
                        bdd_values_sum * grad_phi * normal // (g, n ∂_n v)
                        +
                        prev_grad * normal * phi
                        +
                        prev_value * grad_phi * normal
                        -
                        mu * prev_value * phi
                ) * fe_values.JxW(q);        // ds
            }
        }
        this->rhs.add(loc2glb, local_rhs);
    }


    template<int dim>
    void HeatEqn<dim>::
    compute_condition_number() {
        std::cout << "Compute condition number" << std::endl;

        // Invert the stiffness_matrix
        FullMatrix<double> stiffness_matrix_full(this->solution.size());
        stiffness_matrix_full.copy_from(this->stiffness_matrix);
        FullMatrix<double> inverse(this->solution.size());
        inverse.invert(stiffness_matrix_full);

        double norm = this->stiffness_matrix.frobenius_norm();
        double inverse_norm = inverse.frobenius_norm();

        condition_number = norm * inverse_norm;
        std::cout << "  cond_num = " << condition_number << std::endl;

        // TODO bruk eigenvalues istedet
    }


    template<int dim>
    Error HeatEqn<dim>::
    compute_time_error(std::vector<Error> errors) {
        double l2_error_integral = 0;
        double h1_error_integral = 0;

        double l_inf_l2 = 0;
        double l_inf_h1 = 0;

        for (Error error : errors) {
            l2_error_integral += tau * pow(error.l2_error, 2);
            h1_error_integral += tau * pow(error.h1_semi, 2);

            if (error.l2_error > l_inf_l2)
                l_inf_l2 = error.l2_error;
            if (error.h1_error > l_inf_h1)
                l_inf_h1 = error.h1_error;
        }

        Error error;
        error.h = this->h;
        error.tau = tau;

        error.l2_error = pow(l2_error_integral, 0.5);
        error.h1_error = pow(l2_error_integral + h1_error_integral, 0.5);
        error.h1_semi = pow(h1_error_integral, 0.5);

        error.l_inf_l2_error = l_inf_l2;
        error.l_inf_h1_error = l_inf_h1;
        return error;
    }


    template<int dim>
    void HeatEqn<dim>::
    write_header_to_file(std::ofstream &file) {
        file
                << "h, \\tau, \\|u\\|_{L^2}, \\|u\\|_{H^1}, |u|_{H^1}, \\|u\\|_{l^\\infty L^2}, \\|u\\|_{l^\\infty H^1}, \\kappa(A)"
                << std::endl;
    }


    template<int dim>
    void HeatEqn<dim>::
    write_error_to_file(Error &error, std::ofstream &file) {
        file << error.h << ","
             << error.tau << ","
             << error.l2_error << ","
             << error.h1_error << ","
             << error.h1_semi << ","
             << error.l_inf_l2_error << ","
             << error.l_inf_h1_error << ","
             << error.cond_num << std::endl;
    }


    template<int dim>
    void HeatEqn<dim>::
    write_time_header_to_file(std::ofstream &file) {
        file << "k, \\|u\\|_{L^2}, \\|u\\|_{H^1}, |u|_{H^1}, \\kappa(A)"
             << std::endl;
    }


    template<int dim>
    void HeatEqn<dim>::
    write_time_error_to_file(Error &error, std::ofstream &file) {
        file << error.time_step << ","
             << error.l2_error << ","
             << error.h1_error << ","
             << error.h1_semi << ","
             << error.cond_num << std::endl;
    }


    template
    class HeatEqn<2>;

    template
    class HeatEqn<3>;


} // namespace examples::cut::HeatEquation