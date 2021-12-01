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

#include <cmath>
#include <fstream>

#include "cutfem/nla/sparsity_pattern.h"
#include "cutfem/stabilization/jump_stabilization.h"

#include "poisson.h"


using namespace cutfem;


template<int dim>
Poisson<dim>::Poisson(const double radius,
                      const double half_length,
                      const unsigned int n_refines,
                      const int element_order,
                      const bool write_output,
                      Function<dim> &rhs,
                      Function<dim> &bdd_values,
                      Function<dim> &analytical_soln,
                      Function<dim> &domain_func,
                      const bool stabilized)
        : ScalarProblem<dim>(n_refines, element_order, write_output,
                             domain_func, analytical_soln, stabilized),
          radius(radius), half_length(half_length) {
    // Use no constraints when projecting.
    this->constraints.close();

    this->rhs_function = &rhs;
    this->boundary_values = &bdd_values;
}


template<int dim>
void Poisson<dim>::
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
    void Poisson<dim>::
    pre_matrix_assembly() {
        // Set the CutFEM stabilization scaling constant.
        double gamma_A = 0.5;
        this->stabilization_scaling = gamma_A / (this->h * this->h);
    }


    template<int dim>
    void Poisson<dim>::
    assemble_local_over_cell(
        const FEValues<dim> &fe_values,
        const std::vector<types::global_dof_index> &loc2glb) {
    // TODO generelt: er det for mange hjelpeobjekter som opprettes her i cella?
    //  bør det heller gjøres i funksjonen før og sendes som argumenter? hvis
    //  det er mulig mtp cellene som blir cut da

    // Matrix and vector for the contribution of each cell
    const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);

    // Vector for values of the RightHandSide for all quadrature points on a cell.
    std::vector<double> rhs_values(fe_values.n_quadrature_points);
    this->rhs_function->value_list(fe_values.get_quadrature_points(), rhs_values);

    // Calculate often used terms in the beginning of each cell-loop
    std::vector<double> phi(dofs_per_cell);
    std::vector<Tensor<1, dim>> grad_phi(dofs_per_cell);

    for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
        for (const unsigned int k : fe_values.dof_indices()) {
            grad_phi[k] = fe_values.shape_grad(k, q);
            phi[k] = fe_values.shape_value(k, q);
        }

        for (const unsigned int i : fe_values.dof_indices()) {
            for (const unsigned int j : fe_values.dof_indices()) {
                local_matrix(i, j) += grad_phi[j] * grad_phi[i] *
                                      fe_values.JxW(q); // dx
            }
            // RHS
            local_rhs(i) += rhs_values[q] * phi[i] // (f, v)
                            * fe_values.JxW(q);      // dx
        }
    }
    this->stiffness_matrix.add(loc2glb, local_matrix);
    this->rhs.add(loc2glb, local_rhs);
}


template<int dim>
void
Poisson<dim>::assemble_local_over_surface(
        const FEValuesBase<dim> &fe_values,
        const std::vector<types::global_dof_index> &loc2glb) {
    // Matrix and vector for the contribution of each cell
    const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);

    // Evaluate the boundary function for all quadrature points on this face.
    std::vector<double> bdd_values(fe_values.n_quadrature_points);
    this->boundary_values->value_list(fe_values.get_quadrature_points(), bdd_values);

    // Calculate often used terms in the beginning of each cell-loop
    std::vector<Tensor<1, dim>> grad_phi(dofs_per_cell);
    std::vector<double> phi(dofs_per_cell);

    double gamma = 20 * this->element_order * (this->element_order + 1);
    double mu = gamma / this->h; // Penalty parameter
    Tensor<1, dim> normal;

    for (unsigned int q : fe_values.quadrature_point_indices()) {
        normal = fe_values.normal_vector(q);

        for (const unsigned int k : fe_values.dof_indices()) {
            phi[k] = fe_values.shape_value(k, q);
            grad_phi[k] = fe_values.shape_grad(k, q);
        }

        for (const unsigned int i : fe_values.dof_indices()) {
            for (const unsigned int j : fe_values.dof_indices()) {
                local_matrix(i, j) +=
                        (mu * phi[j] * phi[i]  // mu (u, v)
                         -
                         grad_phi[j] * normal * phi[i] // (∂_n u,v)
                         -
                         phi[j] * grad_phi[i] * normal // (u,∂_n v)
                        ) * fe_values.JxW(q); // ds
            }

            local_rhs(i) +=
                    (mu * bdd_values[q] * phi[i] // mu (g, v)
                     -
                     bdd_values[q] * grad_phi[i] * normal // (g, n ∂_n v)
                    ) * fe_values.JxW(q);        // ds
        }
    }
    this->stiffness_matrix.add(loc2glb, local_matrix);
    this->rhs.add(loc2glb, local_rhs);
}


template<int dim>
void Poisson<dim>::
write_header_to_file(std::ofstream &file) {
    file
            << "h, \\|u\\|_{L^2}, \\|u\\|_{H^1}, |u|_{H^1}, \\|u\\|_{l^\\infty L^2}, \\|u\\|_{l^\\infty H^1}, \\kappa(A)"
            << std::endl;
}


template<int dim>
void Poisson<dim>::
write_error_to_file(ErrorBase* error, std::ofstream &file) {
    auto *err = dynamic_cast<ErrorScalar *>(error);
    file << err->h << ","
         << err->l2_error << ","
         << err->h1_error << ","
         << err->h1_semi << ","
         << err->l_inf_l2_error << ","
         << err->l_inf_h1_error << ","
         << err->cond_num << std::endl;
}


template
class Poisson<2>;

template
class Poisson<3>;


