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

#include <boost/optional.hpp>

#include <cmath>
#include <fstream>

#include "cutfem/geometry/SignedDistanceSphere.h"
#include "cutfem/nla/sparsity_pattern.h"
#include "cutfem/stabilization/jump_stabilization.h"

#include "projection_flow.h"


using namespace cutfem;

namespace examples::cut::projections {

    template<int dim>
    ProjectionFlow<dim>::ProjectionFlow(const double radius,
                                        const double half_length,
                                        const unsigned int n_refines,
                                        const int element_order,
                                        const bool write_output,
                                        LevelSet<dim> &levelset_func,
                                        TensorFunction<1, dim> &analytic_vel,
                                        Function<dim> &analytic_pressure,
                                        const double sphere_radius,
                                        const double sphere_x_coord)
            : FlowProblem<dim>(n_refines, element_order, write_output,
                               levelset_func, analytic_vel, analytic_pressure),
              radius(radius), half_length(half_length) {}


    template<int dim>
    void ProjectionFlow<dim>::
    make_grid(Triangulation<dim> &tria) {
        std::cout << "Creating triangulation" << std::endl;

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
    void ProjectionFlow<dim>::
    assemble_system() {
        std::cout << "Assembling" << std::endl;

        // TODO fjern denne metoden også, og putt i base klassen.
        // Use a helper object to compute the stabilisation for both the velocity
        // and the pressure component.
        // TODO ta ut stabiliseringen i en egen funksjon?
        const FEValuesExtractors::Vector velocities(0);
        stabilization::JumpStabilization<dim, FEValuesExtractors::Vector>
                velocity_stab(*this->dof_handlers.front(),
                              this->mapping_collection,
                              this->cut_mesh_classifier,
                              this->constraints);
        velocity_stab.set_function_describing_faces_to_stabilize(
                stabilization::inside_stabilization);
        velocity_stab.set_weight_function(stabilization::taylor_weights);
        velocity_stab.set_extractor(velocities);

        const FEValuesExtractors::Scalar pressure(dim);
        stabilization::JumpStabilization<dim, FEValuesExtractors::Scalar>
                pressure_stab(*this->dof_handlers.front(),
                              this->mapping_collection,
                              this->cut_mesh_classifier,
                              this->constraints);
        pressure_stab.set_function_describing_faces_to_stabilize(
                stabilization::inside_stabilization);
        pressure_stab.set_weight_function(stabilization::taylor_weights);
        pressure_stab.set_extractor(pressure);

        // TODO sett disse litt ordentlig.
        double beta_0 = 0.1;
        double gamma_M =
                beta_0 * this->element_order * (this->element_order + 1);

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
        QGauss<dim - 1> face_quadrature_formula(this->mixed_fe.degree + 1);
        FEFaceValues<dim> fe_face_values(this->mixed_fe,
                                         face_quadrature_formula,
                                         update_values | update_gradients |
                                         update_quadrature_points |
                                         update_normal_vectors |
                                         update_JxW_values);

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

            if (fe_values_bulk)
                assemble_local_over_cell(*fe_values_bulk, loc2glb);

            // Compute and add the velocity stabilization.
            velocity_stab.compute_stabilization(cell);
            velocity_stab.add_stabilization_to_matrix(
                    gamma_M, this->stiffness_matrix);
        }
    }

    template<int dim>
    void ProjectionFlow<dim>::
    assemble_local_over_cell(const FEValues<dim> &fe_values,
                             const std::vector<types::global_dof_index> &loc2glb) {

        // Matrix and vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // Vector for values of the RightHandSide for all quadrature points on a cell.
        std::vector<Tensor<1, dim>> u_0_values(fe_values.n_quadrature_points,
                                               Tensor<1, dim>());
        this->analytical_velocity->value_list(fe_values.get_quadrature_points(),
                                              u_0_values);

        std::vector<double> p_0_values(fe_values.n_quadrature_points, 0);
        this->analytical_pressure->value_list(fe_values.get_quadrature_points(),
                                              p_0_values);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        // Calculate often used terms in the beginning of each cell-loop
        std::vector<Tensor<1, dim>> phi_u(dofs_per_cell, Tensor<1, dim>());
        std::vector<double> phi_p(dofs_per_cell);

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
            for (const unsigned int k : fe_values.dof_indices()) {
                phi_u[k] = fe_values[velocities].value(k, q);
                phi_p[k] = fe_values[pressure].value(k, q);
            }

            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    local_matrix(i, j) += (phi_u[j] * phi_u[i]  // (u, v)
                                           +
                                           phi_p[j] * phi_p[i]  // (p, q)
                                          ) * fe_values.JxW(q); // dx
                }
                // RHS
                local_rhs(i) += (u_0_values[q] * phi_u[i] // (u_0, v)
                                 +
                                 p_0_values[q] * phi_p[i]
                                ) * fe_values.JxW(q);      // dx
            }
        }
        this->stiffness_matrix.add(loc2glb, local_matrix);
        this->rhs.add(loc2glb, local_rhs);
    }


    template<int dim>
    void ProjectionFlow<dim>::
    assemble_local_over_surface(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {
    }


    template
    class ProjectionFlow<2>;

    template
    class ProjectionFlow<3>;

} // namespace examples::cut::projections
