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
#include <stdexcept>

#include "cutfem/geometry/SignedDistanceSphere.h"
#include "cutfem/nla/sparsity_pattern.h"
#include "cutfem/stabilization/jump_stabilization.h"

#include "../utils/utils.h"

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
                    TensorFunction<1, dim> &bdd_values,
                    TensorFunction<1, dim> &analytic_vel,
                    Function<dim> &analytic_pressure,
                    LevelSet <dim> &levelset_func,
                    const int do_nothing_id,
                    const bool stabilized)
            : StokesEquation::StokesEqn<dim>(nu, tau, radius, half_length, n_refines,
                             element_order, write_output,
                             rhs, bdd_values,
                             analytic_vel, analytic_pressure, levelset_func,
                             do_nothing_id, stabilized, false) {}


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
    assemble_matrix_local_over_cell(
            const FEValues<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {

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


    template
    class NavierStokesEqn<2>;

    template
    class NavierStokesEqn<3>;

} // namespace examples::cut::NavierStokes
