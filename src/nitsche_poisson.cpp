#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

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
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <fstream>
#include <iostream>

using namespace dealii;

template<int dim>
class PoissonNitsche {
public:
    PoissonNitsche(const unsigned int degree);

    void run();

private:
    void make_grid();

    void setup_system();

    void assemble_system();

    void solve();

    void output_results() const;

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;
};

// Functions for right hand side and boundary values.

template<int dim>
class RightHandSide : public Function<dim> {
public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template<int dim>
class BoundaryValues : public Function<dim> {
public:

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
};


template<int dim>
double RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const {
    (void) p;
    return 1;
}

template<int dim>
double BoundaryValues<dim>::value(const Point<dim> &p, const unsigned int) const {
    (void) p;
    return 0;
}


template<int dim>
PoissonNitsche<dim>::PoissonNitsche(const unsigned int degree)
        : fe(degree), dof_handler(triangulation) {}


template<int dim>
void PoissonNitsche<dim>::make_grid() {
    GridGenerator::cylinder(triangulation, 5, 20);
    GridTools::remove_anisotropy(triangulation, 1.618, 5);
    triangulation.refine_global(dim == 2);

    // Write svg of grid to file.
    if (dim == 2) {
        std::ofstream out("poisson-nitsche-grid.svg");
        GridOut grid_out;
        grid_out.write_svg(triangulation, out);
        std::cout << "Grid written to file as svg." << std::endl;
    }

    std::cout << "  Number of active cells: " << triangulation.n_active_cells() << std::endl;

}

template<int dim>
void PoissonNitsche<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);
    std::cout << "  Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

template<int dim>
void PoissonNitsche<dim>::assemble_system() {
    QGauss<dim> quadrature_formula(fe.degree + 1);

    RightHandSide<dim> right_hand_side;
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    cell_matrix(i, j) +=
                            (fe_values.shape_value(i, q_index) *  // phi_i(x_q)
                             fe_values.shape_value(j, q_index) *  // phi_j(x_q)
                             fe_values.JxW(q_index));             // dx
                }


                const auto x_q = fe_values.quadrature_point(q_index);
                cell_rhs(i) += (fe_values.shape_value(i, q_index) *  // phi_i(x_q)
                                right_hand_side.value(x_q) *         // f(x_q)
                                fe_values.JxW(q_index));             // dx
            }
        }

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        for (const unsigned int i : fe_values.dof_indices()) {
            for (const unsigned int j : fe_values.dof_indices()) {
                system_matrix.add(local_dof_indices[i],
                                  local_dof_indices[j],
                                  cell_matrix(i, j));
            }
            system_rhs(local_dof_indices[i]) + cell_rhs(i);
        }
    }
}

template<int dim>
void PoissonNitsche<dim>::solve() {
    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    std::cout << "   " << solver_control.last_step()
              << " CG iterations needed to obtain convergence." << std::endl;
}

template<int dim>
void PoissonNitsche<dim>::output_results() const {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();
    std::ofstream out(dim == 2 ? "solution-2d.vtk" : "solution-3d.vtk");
    data_out.write_vtk(out);
}

template<int dim>
void PoissonNitsche<dim>::run() {
    make_grid();
    setup_system();
    assemble_system();
    solve();
    output_results();
}

int main() {
    std::cout << "PoissonNitsche" << std::endl;
    {
        PoissonNitsche<2> poissonNitsche(1);
        poissonNitsche.run();
    }
}