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

#include <cmath>
#include <fstream>

#include "cutfem/geometry/SignedDistanceSphere.h"
#include "cutfem/nla/sparsity_pattern.h"

#include "navier_stokes.h"


using namespace cutfem;

namespace examples::cut::NavierStokes {

    using namespace utils;
    using namespace examples::cut;

    template<int dim>
    NavierStokesEqn<dim>::
    NavierStokesEqn(const double nu,
                    const double tau,
                    const double radius,
                    const double half_length,
                    const unsigned int n_refines,
                    const int element_order,
                    const bool write_output,
                    TensorFunction<1, dim> &rhs,
                    TensorFunction<1, dim> &conv_field,
                    TensorFunction<1, dim> &bdd_values,
                    TensorFunction<1, dim> &analytic_vel,
                    Function<dim> &analytic_pressure,
                    LevelSet<dim> &levelset_func,
                    const bool semi_implicit,
                    const int do_nothing_id,
                    const bool stabilized)
            : StokesEquation::StokesEqn<dim>(nu, tau, radius, half_length,
                                             n_refines,
                                             element_order, write_output,
                                             rhs, bdd_values,
                                             analytic_vel, analytic_pressure,
                                             levelset_func,
                                             do_nothing_id, stabilized,
                                             false),
              semi_implicit(semi_implicit) {
        convection_field = &conv_field;
    }


    template<int dim>
    void NavierStokesEqn<dim>::
    make_grid(Triangulation<dim> &tria) {
        std::cout << "Creating triangulation" << std::endl;

        GridGenerator::cylinder(tria, this->radius, this->half_length);
        GridTools::remove_anisotropy(tria, 1.618, 5);
        tria.refine_global(this->n_refines);

        this->mapping_collection.push_back(MappingCartesian<dim>());

        // Save the cell-size, we need it in the Nitsche term.
        typename Triangulation<dim>::active_cell_iterator cell =
                tria.begin_active();
        this->h = std::pow(cell->measure(), 1.0 / dim);
    }


    template<int dim>
    void NavierStokesEqn<dim>::
    pre_matrix_assembly() {
        // Set the velocity and pressure stabilization scalings. These can
        // be overridden in a subclass constructor.
        double gamma_u = 0.5;
        double gamma_p = 0.5;

        if (semi_implicit) {
            std::cout << "Stabilization constants set for Navier-Stokes "
                         "(semi-implicit convection term)." << std::endl;
            this->velocity_stab_scaling =
                    gamma_u * (1 + this->tau / this->h +
                               this->tau * this->nu / pow(this->h, 2));
            this->pressure_stab_scaling =
                    -gamma_p * this->tau /
                    (this->nu + this->h + pow(this->h, 2) / this->tau);
        } else {
            std::cout << "Stabilization constants set for Navier-Stokes "
                         "(explicit convection term)." << std::endl;
            this->velocity_stab_scaling =
                    gamma_u * (1 + this->tau * this->nu / pow(this->h, 2));
            this->pressure_stab_scaling =
                    -gamma_p * this->tau /
                    (this->nu + pow(this->h, 2) / this->tau);
        }
    }


    template<int dim>
    void NavierStokesEqn<dim>::
    assemble_matrix_local_over_cell(
            const FEValues<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {
        // Assemble the convection terms.
        if (semi_implicit) {
            if (this->moving_domain) {
                assemble_convection_over_cell_moving_domain(fe_values, loc2glb);
            } else {
                assemble_convection_over_cell(fe_values, loc2glb);
            }
        }
        // The following is identical to the method in the solver for the
        // Stokes equation.

        // Matrix and vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // Vector for values of the RightHandSide for all quadrature points on a cell.
        std::vector<Tensor<1, dim>> rhs_values(fe_values.n_quadrature_points,
                                               Tensor<1, dim>());
        this->rhs_function->value_list(fe_values.get_quadrature_points(),
                                       rhs_values);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        // Calculate often used terms in the beginning of each cell-loop
        std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
        std::vector<double> div_phi_u(dofs_per_cell);
        std::vector<Tensor<1, dim>> phi_u(dofs_per_cell, Tensor<1, dim>());
        std::vector<double> phi_p(dofs_per_cell);

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
            for (const unsigned int k : fe_values.dof_indices()) {
                grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                div_phi_u[k] = fe_values[velocities].divergence(k, q);
                phi_u[k] = fe_values[velocities].value(k, q);
                phi_p[k] = fe_values[pressure].value(k, q);
            }

            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    local_matrix(i, j) +=
                            (this->bdf_coeffs[0]
                             * phi_u[j] * phi_u[i]  // (u, v)
                             +
                             (this->nu * scalar_product(grad_phi_u[j],
                                                        grad_phi_u[i]) // (grad u, grad v)
                              - (div_phi_u[i] * phi_p[j])   // -(div v, p)
                              - (div_phi_u[j] * phi_p[i])   // -(div u, q)
                             ) * this->tau) *
                            fe_values.JxW(q); // dx
                }
                // NB: rhs is assembled in assemble_rhs().
            }
        }
        this->stiffness_matrix.add(loc2glb, local_matrix);
    }


    template<int dim>
    void NavierStokesEqn<dim>::
    assemble_convection_over_cell(
            const FEValues<dim> &fe_v,
            const std::vector<types::global_dof_index> &loc2glb) {
        // Vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_v.get_fe().dofs_per_cell;
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

        // Create vector of the previous solutions values
        std::vector<Tensor<1, dim>> val(fe_v.n_quadrature_points,
                                        Tensor<1, dim>());
        // This vector contains (-, u^n, u^(n-1)) for BDF-2.
        std::vector<std::vector<Tensor<1, dim>>> prev_solution_values(
                this->solutions.size(), val);

        const FEValuesExtractors::Vector v(0);

        // Get the values of the previous solutions, and insert into the
        // vector initialized above.
        for (unsigned long k = 1; k < this->solutions.size(); ++k) {
            fe_v[v].get_function_values(this->solutions[k],
                                        prev_solution_values[k]);
        }

        std::vector<Tensor<1, dim>> conv_field(fe_v.n_quadrature_points,
                                               Tensor<1, dim>());
        this->convection_field->value_list(fe_v.get_quadrature_points(),
                                           conv_field);

        Tensor<1, dim> extrapolation;
        std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
        std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
        for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q) {
            // Compute the extrapolated value by using the previous steps.
            extrapolation = Tensor<1, dim>();
            for (unsigned long k = 1; k < this->solutions.size(); ++k) {
                extrapolation +=
                        this->extrap_coeffs[k] * prev_solution_values[k][q];
            }

            // Compute the gradient values at the dofs, for these quadrature
            // points.
            for (const unsigned int k : fe_v.dof_indices()) {
                grad_phi_u[k] = fe_v[v].gradient(k, q);
                phi_u[k] = fe_v[v].value(k, q);
            }

            // Compute the addition to the local matrix
            for (const unsigned int i : fe_v.dof_indices()) {
                for (const unsigned int j : fe_v.dof_indices()) {
                    // TODO check index!
                    // Assemble the term (u·∇)u = (∇u)u_e, where u_e is the
                    // extrapolated u-value.
                    local_matrix(i, j) +=
                            ((grad_phi_u[j] * conv_field[q]) * phi_u[i]) *
                            this->tau * fe_v.JxW(q);
                }
            }
        }
        this->stiffness_matrix.add(loc2glb, local_matrix);
    }


    template<int dim>
    void NavierStokesEqn<dim>::
    assemble_convection_over_cell_moving_domain(
            const FEValues<dim> &fe_v,
            const std::vector<types::global_dof_index> &loc2glb) {

        // TODO needed?
        const hp::FECollection<dim> &fe_collection = this->dof_handlers.front()->get_fe_collection();
        const hp::QCollection<dim> q_collection(fe_v.get_quadrature());

        // Vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_v.get_fe().dofs_per_cell;
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

        const FEValuesExtractors::Vector v(0);

        // Create vector of the previous solutions values
        std::vector<Tensor<1, dim>> val(fe_v.n_quadrature_points,
                                        Tensor<1, dim>());
        std::vector<std::vector<Tensor<1, dim>>> prev_solution_values(
                this->solutions.size(), val);

        // Get the values of the previous solutions, and insert into the
        // vector initialized above.

        // First get a FEValues object that can do calculations over the
        // current cell, but over the solutions in previous time steps.
        const typename Triangulation<dim>::active_cell_iterator &cell =
                fe_v.get_cell();
        hp::FEValues<dim> hp_fe_values(this->mapping_collection,
                                       fe_collection,
                                       q_collection,
                                       update_values);

        // Read out the solution values from the previous time steps that we
        // need for the BDF-method.
        for (unsigned long k = 1; k < this->solutions.size(); ++k) {
            typename hp::DoFHandler<dim>::active_cell_iterator cell_prev(
                    &(this->triangulation), cell->level(), cell->index(),
                    this->dof_handlers[k].get());
            const FiniteElement<dim> &fe = cell_prev->get_fe();
            if (fe.n_dofs_per_cell() == 0) {
                // This means that in the previous solution step, this cell had
                // FE_Nothing elements. We can therefore not use that cell to
                // get the values we need for the BDF-formula. If this happens
                // then the active mesh in the previous step(s) need to be
                // extended, such that the cells outside the physical domain
                // can be stabilized. When the aftive mesh is sufficiently
                // big in all time steps, we should never enter this clause.
                // If this happens, the values of 0 vill be used.
                std::cout << "# NB: need larger cell buffer outside the "
                             "physical domain, to compute the convection term."
                          << std::endl;
            } else {
                // Get the function values from the previous time steps.
                // TODO check that this is actually done.
                hp_fe_values.reinit(cell_prev);
                const FEValues<dim> &fe_values_prev = hp_fe_values.get_present_fe_values();
                fe_values_prev[v].get_function_values(this->solutions[k],
                                                      prev_solution_values[k]);
            }
        }

        Tensor<1, dim> extrapolation;
        std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
        std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
        for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q) {
            extrapolation = Tensor<1, dim>();
            for (unsigned int k = 1; k < this->solutions.size(); ++k) {
                extrapolation +=
                        this->extrap_coeffs[k] * prev_solution_values[k][q];
            }

            for (const unsigned int k : fe_v.dof_indices()) {
                grad_phi_u[k] = fe_v[v].gradient(k, q);
                phi_u[k] = fe_v[v].value(k, q);
            }

            for (const unsigned int i : fe_v.dof_indices()) {
                for (const unsigned int j : fe_v.dof_indices()) {
                    // Assemble the term (u·∇)u = (∇u)u_e, where u_e is the
                    // extrapolated u-value.
                    local_matrix(i, j) +=
                            ((grad_phi_u[j] * extrapolation) * phi_u[i]) *
                            this->tau * fe_v.JxW(q);
                }
            }
        }
        this->stiffness_matrix.add(loc2glb, local_matrix);
    }


    template<int dim>
    void NavierStokesEqn<dim>::
    assemble_rhs(int time_step) {
        std::cout << "Assembling rhs: Navier-Stokes" << std::endl;

        NonMatching::RegionUpdateFlags region_update_flags;
        region_update_flags.inside = update_values | update_quadrature_points
                                     | update_gradients | update_JxW_values;
        region_update_flags.surface =
                update_values | update_JxW_values |
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
        QGauss<dim - 1> face_quadrature_formula(this->mixed_fe.degree + 1);
        FEFaceValues<dim> fe_face_values(this->mixed_fe,
                                         face_quadrature_formula,
                                         update_values | update_gradients |
                                         update_quadrature_points |
                                         update_normal_vectors |
                                         update_JxW_values);

        // TODO setter dette alle elementene i rhs til 0?
        // rhs = 0;

        for (const auto &cell : this->dof_handlers.front()->active_cell_iterators()) {
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
                if (this->moving_domain) {
                    this->assemble_rhs_and_bdf_terms_local_over_cell_moving_domain(
                            *fe_values_bulk, loc2glb);
                } else {
                    this->assemble_rhs_and_bdf_terms_local_over_cell(
                            *fe_values_bulk, loc2glb);
                }
            }

            // Loop through all faces that constitutes the outer boundary of the
            // domain.
            for (const auto &face : cell->face_iterators()) {
                if (face->at_boundary() &&
                    face->boundary_id() != this->do_nothing_id) {
                    fe_face_values.reinit(cell, face);
                    this->assemble_rhs_local_over_surface(fe_face_values,
                                                          loc2glb);
                }
            }

            // Retrieve an FEValues object with quadrature points
            // on the immersed surface.
            const boost::optional<const FEImmersedSurfaceValues<dim> &>
                    fe_values_surface = cut_fe_values.get_surface_fe_values();

            if (fe_values_surface) {
                this->assemble_rhs_local_over_surface(*fe_values_surface,
                                                      loc2glb);
            }
        }
    }


    template<int dim>
    void NavierStokesEqn<dim>::
    assemble_rhs_and_bdf_terms_local_over_cell(
            const FEValues<dim> &fe_v,
            const std::vector<types::global_dof_index> &loc2glb) {
        // Vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_v.get_fe().dofs_per_cell;
        Vector<double> local_rhs(dofs_per_cell);

        // Vector for values of the RightHandSide for all quadrature points on a cell.
        std::vector<Tensor<1, dim>> rhs_values(fe_v.n_quadrature_points,
                                               Tensor<1, dim>());
        this->rhs_function->value_list(fe_v.get_quadrature_points(),
                                       rhs_values);

        const FEValuesExtractors::Vector v(0);

        // Get the solution values from the previous steps.
        std::vector<Tensor<1, dim>> val(fe_v.n_quadrature_points,
                                        Tensor<1, dim>());
        std::vector<std::vector<Tensor<1, dim >>> prev_values(
                this->solutions.size(), val);

        // Get the solution gradients from the previous steps.
        std::vector<Tensor<2, dim>> grad_val(fe_v.n_quadrature_points,
                                             Tensor<2, dim>());
        std::vector<std::vector<Tensor<2, dim>>> prev_gradients(
                this->solutions.size(), grad_val);

        for (unsigned int k = 1; k < this->solutions.size(); ++k) {
            fe_v[v].get_function_values(this->solutions[k], prev_values[k]);
            if (!semi_implicit) {
                // Then the convection terms should be assembled explicitly,
                // using the extrapolation from earlier steps, matching the
                // order of the chosen BDF method.
                fe_v[v].get_function_gradients(this->solutions[k],
                                               prev_gradients[k]);
            }
        }

        Tensor<1, dim> extrap;
        Tensor<2, dim> grad_extrap;
        Tensor<1, dim> phi_u;
        Tensor<1, dim> bdf_terms;

        for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q) {
            // Compute the extrapolated value by using the previous steps.
            extrap = Tensor<1, dim>();
            grad_extrap = Tensor<2, dim>();

            if (!semi_implicit) {
                for (unsigned long k = 1; k < this->solutions.size(); ++k) {
                    // Only needed for explicit assembly of the convection term.
                    // If semi_implicit, grad_extrap remains zero, and the term
                    // vanishes from expression below.
                    extrap += this->extrap_coeffs[k] * prev_values[k][q];
                    grad_extrap +=
                            this->extrap_coeffs[k] * prev_gradients[k][q];
                }
            }

            bdf_terms = Tensor<1, dim>();
            for (unsigned int k = 1; k < this->solutions.size(); ++k) {
                bdf_terms += this->bdf_coeffs[k] * prev_values[k][q];
            }
            for (const unsigned int i : fe_v.dof_indices()) {
                phi_u = fe_v[v].value(i, q);
                local_rhs(i) += (this->tau * rhs_values[q] * phi_u // τ(f, v)
                                 - bdf_terms * phi_u               // BDF-terms
                                 - (grad_extrap * extrap)          // convection
                                   * phi_u * this->tau
                                ) * fe_v.JxW(q);                   // dx
            }
        }
        this->rhs.add(loc2glb, local_rhs);
    }


    template<int dim>
    void NavierStokesEqn<dim>::
    assemble_rhs_and_bdf_terms_local_over_cell_moving_domain(
            const FEValues<dim> &fe_v,
            const std::vector<types::global_dof_index> &loc2glb) {

        // TODO needed?
        const hp::FECollection<dim> &fe_collection = this->dof_handlers.front()->get_fe_collection();
        const hp::QCollection<dim> q_collection(fe_v.get_quadrature());

        // Vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_v.get_fe().dofs_per_cell;
        Vector<double> local_rhs(dofs_per_cell);

        const FEValuesExtractors::Vector v(0);

        // Vector for values of the RightHandSide for all quadrature points on a cell.
        std::vector<Tensor<1, dim>> rhs_values(fe_v.n_quadrature_points,
                                               Tensor<1, dim>());
        this->rhs_function->value_list(fe_v.get_quadrature_points(),
                                       rhs_values);

        // Create vector of the previous solutions values
        std::vector<Tensor<1, dim>> val(fe_v.n_quadrature_points,
                                        Tensor<1, dim>());
        std::vector<std::vector<Tensor<1, dim>>> prev_values(
                this->solutions.size(), val);

        // Store the solution gradients from the previous steps. This is only
        // needed if the convection term is assembled explicitly.
        std::vector<Tensor<2, dim>> grad_val(fe_v.n_quadrature_points,
                                             Tensor<2, dim>());
        std::vector<std::vector<Tensor<2, dim>>> prev_gradients(
                this->solutions.size(), grad_val);

        // The the values of the previous solutions, and insert into the
        // matrix initialized above.
        const typename Triangulation<dim>::active_cell_iterator &cell =
                fe_v.get_cell();
        hp::FEValues<dim> hp_fe_values(this->mapping_collection,
                                       fe_collection,
                                       q_collection,
                                       update_values |
                                       update_gradients);

        // Read out the solution values from the previous time steps that we
        // need for the BDF-method
        for (unsigned long k = 1; k < this->solutions.size(); ++k) {
            typename hp::DoFHandler<dim>::active_cell_iterator cell_prev(
                    &(this->triangulation), cell->level(), cell->index(),
                    this->dof_handlers[k].get());
            const FiniteElement<dim> &fe = cell_prev->get_fe();
            if (fe.n_dofs_per_cell() == 0) {
                // This means that in the previous solution step, this cell had
                // FE_Nothing elements. We can therefore not use that cell to
                // get the values we need for the BDF-formula. If this happens
                // then the active mesh in the previous step(s) need to be
                // extended, such that the cells outside the physical domain
                // can be stabilized. When the aftive mesh is sufficiently
                // big in all time steps, we should never enter this clause.
                // If this happens, the values of 0 vill be used.
                std::cout << "# NB: need larger cell buffer outside the "
                             "physical domain." << std::endl;
            } else {
                // Get the function values from the previous time steps.
                hp_fe_values.reinit(cell_prev);
                const FEValues<dim> &fe_values_prev = hp_fe_values.get_present_fe_values();
                fe_values_prev[v].get_function_values(this->solutions[k],
                                                      prev_values[k]);
                if (!semi_implicit) {
                    // Only needed when the convection term is assembled
                    // exolicitly.
                    fe_values_prev[v].get_function_gradients(this->solutions[k],
                                                             prev_gradients[k]);
                }
            }
        }

        Tensor<1, dim> extrap;
        Tensor<2, dim> grad_extrap;
        Tensor<1, dim> phi_u;
        Tensor<1, dim> bdf_terms;

        for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q) {
            // Compute the extrapolated value by using the previous steps.
            extrap = Tensor<1, dim>();
            grad_extrap = Tensor<2, dim>();

            if (!semi_implicit) {
                for (unsigned long k = 1; k < this->solutions.size(); ++k) {
                    // Only needed for explicit assembly of the convection term.
                    // If semi_implicit, grad_extrap remains zero, and the term
                    // vanishes from expression below.
                    extrap += this->extrap_coeffs[k] * prev_values[k][q];
                    grad_extrap +=
                            this->extrap_coeffs[k] * prev_gradients[k][q];
                }
            }

            bdf_terms = 0;
            for (unsigned long k = 1; k < this->solutions.size(); ++k) {
                bdf_terms += this->bdf_coeffs[k] * prev_values[k][q];
            }
            for (const unsigned int i : fe_v.dof_indices()) {
                phi_u = fe_v[v].value(i, q);
                local_rhs(i) += (this->tau * rhs_values[q] * phi_u // τ(f, v)
                                 - bdf_terms * phi_u               // BDF-terms
                                 - (grad_extrap * extrap)
                                   * phi_u * this->tau             // convection
                                ) * fe_v.JxW(q);                   // dx
            }
        }
        this->rhs.add(loc2glb, local_rhs);
    }


    template
    class NavierStokesEqn<2>;

    template
    class NavierStokesEqn<3>;

} // namespace examples::cut::NavierStokes
