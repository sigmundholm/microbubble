#include <deal.II/base/data_out_base.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/std_cxx17/optional.h>

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
#include <deal.II/non_matching/fe_immersed_values.h>

#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>

#include <assert.h>
#include <cmath>
#include <fstream>

#include "../utils/stabilization/jump_stabilization.h"

#include "../utils/utils.h"

#include "heat_eqn.h"


using namespace cutfem;


namespace examples::cut::HeatEquation {

    using NonMatching::FEImmersedSurfaceValues;
    using namespace utils;

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
                          LevelSet<dim> &levelset_func,
                          const bool stabilized,
                          const bool crank_nicholson,
                          const bool compute_error)
            : ScalarProblem<dim>(n_refines, element_order, write_output,
                                 levelset_func, analytical_soln, stabilized,
                                 false, compute_error),
              nu(nu), radius(radius), half_length(half_length) {
        this->tau = tau;
        this->crank_nicholson = crank_nicholson;

        this->rhs_function = &rhs;
        this->boundary_values = &bdd_values;
    }


    template<int dim>
    void HeatEqn<dim>::
    set_function_times(double time) {
        this->rhs_function->set_time(time);
        this->boundary_values->set_time(time);
        this->analytical_solution->set_time(time);
        this->levelset_function->set_time(time);
    }


    template<int dim>
    void
    HeatEqn<dim>::make_grid(Triangulation<dim> &tria) {
        this->pcout << "Creating triangulation" << std::endl;

        GridGenerator::cylinder(tria, radius, half_length);
        GridTools::remove_anisotropy(tria, 1.618, 5);
        tria.refine_global(this->n_refines);

        this->mapping_collection.push_back(MappingCartesian<dim>());
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
        this->pcout << "Assembling" << std::endl;

        this->stiffness_matrix = 0;
        this->rhs = 0;

        // Use a helper object to compute the stabilisation for both the velocity
        // and the pressure component.
        stabilization::JumpStabilization<dim, FEValuesExtractors::Scalar>
                stabilization(*(this->dof_handlers.front()),
                              this->mapping_collection,
                              this->cut_mesh_classifier,
                              this->constraints);
        if (this->stabilized) {
            // Object deciding what faces that should be stabilized.
            std::shared_ptr<Selector<dim>> face_selector(
                    new Selector<dim>(this->cut_mesh_classifier));

            stabilization.set_faces_to_stabilize(face_selector);
            stabilization.set_weight_function(stabilization::taylor_weights);
            const FEValuesExtractors::Scalar velocities(0);
            stabilization.set_extractor(velocities);
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

        for (const auto &cell : this->dof_handlers.front()->active_cell_iterators()) {
            if (cell->is_locally_owned()) {
                const unsigned int n_dofs = cell->get_fe().dofs_per_cell;

                const LocationToLevelSet location =
                        this->cut_mesh_classifier.location_to_level_set(cell);

                std::vector<types::global_dof_index> loc2glb(n_dofs);
                cell->get_dof_indices(loc2glb);

                // This call will compute quadrature rules relevant for this cell
                // in the background.
                cut_fe_values.reinit(cell);

                if (location != LocationToLevelSet::outside) {

                    // Retrieve an FEValues object with quadrature points
                    // over the full cell.
                    const std_cxx17::optional<FEValues<dim>>& fe_values_bulk =
                            cut_fe_values.get_inside_fe_values();
                    if (fe_values_bulk) {
                        assemble_matrix_local_over_cell(*fe_values_bulk, loc2glb);
                    }

                    // Retrieve an FEValues object with quadrature points
                    // on the immersed surface.
                    const std_cxx17::optional<FEImmersedSurfaceValues<dim>>&
                            fe_values_surface = cut_fe_values.get_surface_fe_values();
                    if (fe_values_surface) {
                        assemble_matrix_local_over_surface(*fe_values_surface,
                                                        loc2glb);
                    }
                }

                if (this->stabilized) {
                    // Compute and add the velocity stabilization.
                    stabilization.compute_stabilization(cell);
                    double scaling = this->tau * gamma_M +
                                    this->tau * nu * gamma_A / pow(this->h, 2);
                    stabilization.add_stabilization_to_matrix(
                            scaling,
                            this->stiffness_matrix);
                }
            }
        }
        this->stiffness_matrix.compress(VectorOperation::add);
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

        double cn_factor = this->crank_nicholson ? 0.5 : 1;

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
            for (const unsigned int k : fe_values.dof_indices()) {
                grad_phi[k] = fe_values.shape_grad(k, q);
                phi[k] = fe_values.shape_value(k, q);
            }

            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    local_matrix(i, j) += (this->bdf_coeffs[0]
                                           * phi[j] * phi[i]
                                           +
                                           cn_factor * this->tau * nu *
                                           grad_phi[j] *
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
        double cn_factor = this->crank_nicholson ? 0.5 : 1;

        for (unsigned int q : fe_values.quadrature_point_indices()) {
            normal = fe_values.normal_vector(q);

            for (const unsigned int k : fe_values.dof_indices()) {
                phi[k] = fe_values.shape_value(k, q);
                grad_phi[k] = fe_values.shape_grad(k, q);
            }

            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    local_matrix(i, j) +=
                            cn_factor * this->tau * nu * (
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
        this->pcout << "Assembling RHS: Heat Equation" << std::endl;

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

        for (const auto &cell : this->dof_handlers.front()->active_cell_iterators()) {
            if (cell->is_locally_owned()) {
                const unsigned int n_dofs = cell->get_fe().dofs_per_cell;
                std::vector<types::global_dof_index> loc2glb(n_dofs);
                cell->get_dof_indices(loc2glb);

                // This call will compute quadrature rules relevant for this cell
                // in the background.
                cut_fe_values.reinit(cell);

                // Retrieve an FEValues object with quadrature points
                // over the full cell.
                const std_cxx17::optional<FEValues<dim>>& fe_values_bulk =
                        cut_fe_values.get_inside_fe_values();

                if (fe_values_bulk) {
                    if (this->crank_nicholson) {
                        assemble_rhs_local_over_cell_cn(*fe_values_bulk, loc2glb,
                                                        time_step);
                    } else {
                        if (this->moving_domain) {
                            this->assemble_rhs_and_bdf_terms_local_over_cell_moving_domain(
                                    *fe_values_bulk, loc2glb);
                        } else {
                            this->assemble_rhs_and_bdf_terms_local_over_cell(
                                    *fe_values_bulk, loc2glb);

                        }
                    }
                }

                // Retrieve an FEValues object with quadrature points
                // on the immersed surface.
                const std_cxx17::optional<FEImmersedSurfaceValues<dim>>&
                        fe_values_surface = cut_fe_values.get_surface_fe_values();

                if (fe_values_surface) {
                    if (this->crank_nicholson) {
                        assemble_rhs_local_over_surface_cn(*fe_values_surface,
                                                        loc2glb, time_step);
                    } else {
                        assemble_rhs_local_over_surface(*fe_values_surface,
                                                        loc2glb);
                    }
                }
            }
        }
        this->rhs.compress(VectorOperation::add);
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
        assert(this->solutions.size() == 2 && this->bdf_coeffs.size() == 2);

        // Matrix and vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
        Vector<double> local_rhs(dofs_per_cell);

        // Vector for values of the RightHandSide for all quadrature points on a cell.
        std::vector<double> rhs_values(fe_values.n_quadrature_points);
        this->rhs_function->set_time(time_step * this->tau);
        this->rhs_function->value_list(fe_values.get_quadrature_points(),
                                       rhs_values);

        // Compute the rhs values from the previous time step.
        this->rhs_function->set_time((time_step - 1) * this->tau);
        std::vector<double> rhs_values_prev(fe_values.n_quadrature_points);
        this->rhs_function->value_list(fe_values.get_quadrature_points(),
                                       rhs_values_prev);

        // Get the previous solution values.
        std::vector<double> prev_solution_values(fe_values.n_quadrature_points,
                                                 0);

        // TODO riktig solutions index??
        fe_values.get_function_values(this->solutions[1], prev_solution_values);

        // Get the previous solution gradients.
        std::vector<Tensor<1, dim>> prev_solution_grads(
                fe_values.n_quadrature_points, Tensor<1, dim>());
        // TODO samme her
        fe_values.get_function_gradients(this->solutions[1],
                                         prev_solution_grads);

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


                local_rhs(i) += (0.5 * this->tau * (
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
                        this->tau * nu *
                        (mu * bdd_values[q] * phi[i] // mu (g, v)
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

        // this->pcout << "  rhs surf k = " << time_step << std::endl;

        // Evaluate the boundary function for all quadrature points on this face.
        std::vector<double> bdd_values(fe_values.n_quadrature_points);
        this->boundary_values->set_time(time_step * this->tau);
        this->boundary_values->value_list(fe_values.get_quadrature_points(),
                                          bdd_values);

        std::vector<double> bdd_values_prev(fe_values.n_quadrature_points);
        this->boundary_values->set_time((time_step - 1) * this->tau);
        this->boundary_values->value_list(fe_values.get_quadrature_points(),
                                          bdd_values_prev);

        // Get the previous solution values.
        std::vector<double> prev_solution_values(fe_values.n_quadrature_points,
                                                 0);
        fe_values.get_function_values(this->solutions[1], prev_solution_values);

        // Get the previous solution gradients.
        std::vector<Tensor<1, dim>> prev_solution_grads(
                fe_values.n_quadrature_points, Tensor<1, dim>());
        fe_values.get_function_gradients(this->solutions[1],
                                         prev_solution_grads);

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

                local_rhs(i) += 0.5 * this->tau * nu * (
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
    write_header_to_file(std::ofstream &file) {
        if (this->this_mpi_process == 0) {
            file << "h, \\tau, \\|u\\|_{L^2}, \\|u\\|_{H^1}, |u|_{H^1}, "
                    "\\|u\\|_{l^\\infty L^2}, \\|u\\|_{l^\\infty H^1}, \\kappa(A)"
                << std::endl;
        }
    }


    template<int dim>
    void HeatEqn<dim>::
    write_error_to_file(ErrorBase *error, std::ofstream &file) {
        if (this->this_mpi_process == 0) {
            auto *err = dynamic_cast<ErrorScalar *>(error);
            file << err->h << ","
                << err->tau << ","
                << err->l2_error << ","
                << err->h1_error << ","
                << err->h1_semi << ","
                << err->l_inf_l2_error << ","
                << err->l_inf_h1_error << ","
                << err->cond_num << std::endl;
        }
    }


    template
    class HeatEqn<2>;

    template
    class HeatEqn<3>;


} // namespace examples::cut::HeatEquation