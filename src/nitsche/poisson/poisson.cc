#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>

#include <fstream>
#include <iostream>

#include "poisson.h"
#include "rhs.h"


template<int dim>
PoissonNitsche<dim>::PoissonNitsche(const unsigned int degree,
                                    const unsigned int n_refines)
        : degree(degree), n_refines(n_refines), fe(degree),
          dof_handler(triangulation) {}


template<int dim>
void PoissonNitsche<dim>::make_grid() {
    GridGenerator::cylinder(triangulation, 1, 1);
    GridTools::remove_anisotropy(triangulation, 1.618, 5);

    triangulation.refine_global(n_refines);

    // Write svg of grid to file.
    if (dim == 2) {
        std::ofstream out("poisson-nitsche-grid.svg");
        GridOut grid_out;
        grid_out.write_svg(triangulation, out);
        std::cout << "Grid written to file as svg." << std::endl;
    }

    std::cout << "  Number of active cells: " << triangulation.n_active_cells()
              << std::endl;

}

template<int dim>
void PoissonNitsche<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);
    std::cout << "  Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    DoFRenumbering::Cuthill_McKee(dof_handler);
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

template<int dim>
void PoissonNitsche<dim>::assemble_system() {
    QGauss<dim> quadrature_formula(fe.degree + 1);
    QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    RightHandSide<dim> right_hand_side;
    BoundaryValues<dim> boundary_values;

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_v(fe,
                                face_quadrature_formula,
                                update_values | update_gradients |
                                update_quadrature_points |
                                update_normal_vectors |
                                update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<double> phi(dofs_per_cell);
    std::vector<Tensor<1, dim>> grad_phi(dofs_per_cell);
    std::vector<double> g_values(fe_face_v.n_quadrature_points);
    Tensor<1, dim> normal;

    for (const auto &cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        // Integrate the contribution from the interior of each cell
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    cell_matrix(i, j) +=
                            fe_values.shape_grad(j, q_index) *  // u
                            fe_values.shape_grad(i, q_index) *  // v
                            fe_values.JxW(q_index);             // dx
                }

                // RHS
                const auto x_q = fe_values.quadrature_point(q_index);
                cell_rhs(i) += (fe_values.shape_value(i, q_index) *  // v
                                right_hand_side.value(x_q) *         // f(x_q)
                                fe_values.JxW(q_index));             // dx
            }
        }

        double h_local;
        double mu;

        for (const auto &face : cell->face_iterators()) {
            if (face->at_boundary()) {
                fe_face_v.reinit(cell, face);

                h_local = std::pow(face->measure(), 1.0 / (dim - 1));
                double gamma_0 = 10;
                mu = gamma_0 * degree * (degree + 1) /
                     h_local;  // Penalty parameter

                if (h_local > h) {
                    h = h_local;
                }

                for (unsigned int q : fe_face_v.quadrature_point_indices()) {
                    Point<dim> x_q = fe_face_v.quadrature_point(q);
                    double g_q = boundary_values.value(x_q);

                    for (unsigned int k = 0; k < dofs_per_cell; ++k) {
                        phi[k] = fe_face_v.shape_value(k, q);
                        grad_phi[k] = fe_face_v.shape_grad(k, q);
                    }
                    normal = fe_face_v.normal_vector(q);

                    for (const unsigned int i : fe_face_v.dof_indices()) {
                        for (const unsigned int j : fe_face_v.dof_indices()) {

                            cell_matrix(i, j) +=
                                    ((mu * phi[j] * phi[i]  // mu (u, v)
                                      -
                                      normal * grad_phi[j] * phi[i] // (∂_n u,v)
                                      -
                                      phi[j] * normal * grad_phi[i] // (u,∂_n v)
                                     ) * fe_face_v.JxW(q));
                        }

                        cell_rhs(i) +=
                                (mu * g_q * phi[i] // mu (g, v)
                                 -
                                 g_q * normal * grad_phi[i] // (g, n ∂_n v)
                                ) * fe_face_v.JxW(q);        // dx
                    }
                }
            }
        }

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                system_matrix.add(local_dof_indices[i],
                                  local_dof_indices[j],
                                  cell_matrix(i, j));
            }
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }
}

template<int dim>
void PoissonNitsche<dim>::solve() {
    std::cout << "Solving system" << std::endl;
    SparseDirectUMFPACK inverse;
    inverse.initialize(system_matrix);
    inverse.vmult(solution, system_rhs);
}

template<int dim>
void PoissonNitsche<dim>::output_results() const {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();
    std::ofstream out("solution-d" + std::to_string(dim)
                      + "o" + std::to_string(degree)
                      + "r" + std::to_string(n_refines) + ".vtk");
    data_out.write_vtk(out);
}

template<int dim>
Error PoissonNitsche<dim>::compute_error() {
    QGauss<dim> quad(fe.degree + 1);

    FEValues<dim> fe_v(fe, quad,
                       update_values | update_gradients |
                       update_quadrature_points | update_JxW_values);

    double l2_error_integral = 0;
    double h1_semi_error_integral = 0;

    std::vector<double> solution_values(fe_v.n_quadrature_points);
    std::vector<Tensor<1, dim>> solution_gradients(fe_v.n_quadrature_points);

    std::vector<double> exact_values(fe_v.n_quadrature_points);
    std::vector<Tensor<1, dim>> exact_gradients(fe_v.n_quadrature_points);

    AnalyticalSolution<dim> analytical_solution;

    double diff_value;
    Tensor<1, dim> diff_grad;

    for (const auto &cell : dof_handler.active_cell_iterators()) {
        fe_v.reinit(cell);

        fe_v.get_function_values(solution, solution_values);
        fe_v.get_function_gradients(solution, solution_gradients);

        analytical_solution.value_list(fe_v.get_quadrature_points(),
                                       exact_values);
        analytical_solution.gradient_list(fe_v.get_quadrature_points(),
                                          exact_gradients);


        for (const unsigned int q : fe_v.quadrature_point_indices()) {
            diff_value = exact_values[q] - solution_values[q];
            diff_grad = exact_gradients[q] - solution_gradients[q];

            l2_error_integral += diff_value * diff_value * fe_v.JxW(q);
            h1_semi_error_integral += diff_grad * diff_grad * fe_v.JxW(q);
        }
    }

    Error error;
    error.mesh_size = h;
    error.l2_error = pow(l2_error_integral, 0.5);
    error.h1_semi = pow(h1_semi_error_integral, 0.5);
    error.h1_error = pow(l2_error_integral + h1_semi_error_integral, 0.5);
    return error;
}

template<int dim>
Error PoissonNitsche<dim>::run() {
    make_grid();
    setup_system();
    assemble_system();
    solve();
    output_results();
    return compute_error();
}


template<int dim>
void PoissonNitsche<dim>::
write_header_to_file(std::ofstream &file) {
    file << "h, \\|u\\|_{L^2}, \\|u\\|_{H^1}, |u|_{H^1}" << std::endl;
}


template<int dim>
void PoissonNitsche<dim>::
write_error_to_file(Error &error, std::ofstream &file) {
    file << error.mesh_size << ","
         << error.l2_error << ","
         << error.h1_error << ","
         << error.h1_semi << std::endl;
}


template
class PoissonNitsche<2>;

template
class PoissonNitsche<3>;
