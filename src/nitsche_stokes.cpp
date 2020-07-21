#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <fstream>
#include <iostream>

namespace Stokes {


    using namespace dealii;

    template<int dim>
    class StokesNitsche {
    public:
        StokesNitsche(const unsigned int degree);

        void run();

    private:
        void make_grid();

        void setup_dofs();

        void assemble_system();

        void solve();

        void output_results() const;

        const unsigned int degree;
        Triangulation<dim> triangulation;
        FESystem<dim> fe;
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

        virtual void vector_value(const Point<dim> &p, Vector<double> &value) const override;
    };

    template<int dim>
    double RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const {
        (void) p;
        return 0;
    }

    template<int dim>
    void RightHandSide<dim>::vector_value(const Point<dim> &p, Vector<double> &value) const {
        for (unsigned int i = 0; i < value.size(); ++i)
            value[i] = this->value(p, i);
    }


    template<int dim>
    class BoundaryValues : public Function<dim> {
    public:
        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;

        virtual void vector_value(const Point<dim> &p, Vector<double> &value) const override;
    };

    template<int dim>
    double BoundaryValues<dim>::value(const Point<dim> &p, const unsigned int component) const {
        (void) p;
        if (component == 0 && p[0] == 0) {
            if (dim == 2) {
                return -2.5 * (p[1] - 0.41) * p[1];
            }
            throw std::exception(); // TODO fix 3D
        }
        return 0;
    }

    template<int dim>
    void BoundaryValues<dim>::vector_value(const Point<dim> &p, Vector<double> &value) const {
        for (unsigned int c = 0; c < this->n_components; ++c)
            value(c) = BoundaryValues<dim>::value(p, c);
    }

    template<int dim>
    StokesNitsche<dim>::StokesNitsche(const unsigned int degree)
            : degree(degree),
              fe(FESystem<dim>(FE_Q<dim>(degree + 1), dim), 1, FE_Q<dim>(degree),
                 1), // u (with dim components), p (scalar component)
              dof_handler(triangulation) {}
    // TODO noe spesielt for triangulation?

    template<int dim>
    void StokesNitsche<dim>::make_grid() {
        GridGenerator::channel_with_cylinder(triangulation, 0.03, 2, 2.0, true);
        triangulation.refine_global(dim == 2 ? 1 : 0);

        // Write svg of grid to file.
        if (dim == 2) {
            std::ofstream out("nitsche-stokes-grid.svg");
            GridOut grid_out;
            grid_out.write_svg(triangulation, out);
            std::cout << "  Grid written to file as svg." << std::endl;
        }
        std::ofstream out_vtk("nitsche-stokes-grid.vtk");
        GridOut grid_out;
        grid_out.write_vtk(triangulation, out_vtk);
        std::cout << "  Grid written to file as vtk." << std::endl;

        std::cout << "  Number of active cells: " << triangulation.n_active_cells() << std::endl;
    }

    template<int dim>
    void StokesNitsche<dim>::setup_dofs() {
        dof_handler.distribute_dofs(fe);
        DoFRenumbering::Cuthill_McKee(dof_handler);
        DoFRenumbering::component_wise(dof_handler);

        const std::vector<types::global_dof_index> dofs_per_block =
                DoFTools::count_dofs_per_fe_block(dof_handler);
        const unsigned int n_u = dofs_per_block[0];
        const unsigned int n_p = dofs_per_block[1];
        std::cout << "  Number of active cells: " << triangulation.n_active_cells() << std::endl
                  << "  Number of degrees of freedom: " << dof_handler.n_dofs()
                  << " (" << n_u << " + " << n_p << ')' << std::endl;

        DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp);
        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit(sparsity_pattern);

        solution.reinit(dof_handler.n_dofs());
        system_rhs.reinit(dof_handler.n_dofs());
    }

    template<int dim>
    void StokesNitsche<dim>::assemble_system() {

        system_matrix = 0;
        system_rhs = 0;

        QGauss<dim> quadrature_formula(fe.degree + 2);  // TODO degree+1 eller +2?
        QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

        RightHandSide<dim> right_hand_side;
        BoundaryValues<dim> boundary_values;

        FEValues<dim> fe_values(fe,
                                quadrature_formula,
                                update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);
        FEFaceValues<dim> fe_face_values(fe,
                                         face_quadrature_formula,
                                         update_values | update_gradients |
                                         update_quadrature_points | update_normal_vectors |
                                         update_JxW_values);

        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_q_face_points = face_quadrature_formula.size();

        // Matrix and vector for the contribution of each cell
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // Vector for values of the RightHandSide for all quadrature points on a cell.
        // TODO dim eller dim + 1? (se step-22)
        std::vector<Vector<double>>
                rhs_values(n_q_points, Vector<double>(dim));
        std::vector<Vector<double>>
                bdd_values(n_q_face_points, Vector<double>(dim));

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        // Calculate often used terms in the beginning of each cell-loop
        std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
        std::vector<double> div_phi_u(dofs_per_cell);
        std::vector<Tensor<1, dim>> phi_u(dofs_per_cell, Tensor<1, dim>());
        std::vector<double> phi_p(dofs_per_cell);

        double h;
        double mu;
        Tensor<1, dim> normal;
        Tensor<1, dim> x_q;

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);
            local_matrix = 0;
            local_rhs = 0;

            // Get the values for the RightHandSide for all quadrature points in this cell.
            right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_values);

            // Integrate the contribution for each cell
            for (const unsigned int q : fe_values.quadrature_point_indices()) {

                for (const unsigned int k : fe_values.dof_indices()) {
                    grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                    div_phi_u[k] = fe_values[velocities].divergence(k, q);
                    phi_u[k] = fe_values[velocities].value(k, q);
                    phi_p[k] = fe_values[pressure].value(k, q);
                }

                for (const unsigned int i : fe_values.dof_indices()) {
                    for (const unsigned int j : fe_values.dof_indices()) {
                        local_matrix(i, j) +=
                                (scalar_product(grad_phi_u[i],
                                                grad_phi_u[j])  // (grad u, grad v) TODO riktig?
                                 - (div_phi_u[j] * phi_p[i])    // -(div v, p)
                                 - (div_phi_u[i] * phi_p[j])    // -(div u, q)
                                ) * fe_values.JxW(q);           // dx
                    }
                    // RHS
                    // TODO må finnes en oneliner for Vector(dim) * Tensor<1, dim>...
                    // eller skal jeg gjøre som i step-22, er det en primitive?
                    // Calculate the inner product "manually" because one is a Vector and the other a Tensor.
                    double cell_ips = 0;
                    for (unsigned int k = 0; k < dim; ++k) {
                        cell_ips += rhs_values[q](k) * phi_u[q][k];  // (f, v)
                    }
                    local_rhs(i) +=
                            cell_ips             // mu (f(x_q), v(x_q))
                            * fe_values.JxW(q);  // dx
                }
            }


            for (const auto &face : cell->face_iterators()) {

                // TODO hva skal boundary id være?
                if (face->at_boundary() && face->boundary_id() != 1) {
                    fe_face_values.reinit(cell, face);

                    // Evaluate the boundary function for all quadrature points on this face.
                    boundary_values.vector_value_list(fe_face_values.get_quadrature_points(), bdd_values);

                    h = std::pow(face->measure(), 1.0 / (dim - 1));
                    mu = 5 / h;  // Penalty parameter

                    for (unsigned int q : fe_face_values.quadrature_point_indices()) {
                        x_q = fe_face_values.quadrature_point(q);
                        normal = fe_face_values.normal_vector(q);

                        for (const unsigned int k : fe_values.dof_indices()) {
                            grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                            div_phi_u[k] = fe_values[velocities].divergence(k, q);
                            phi_u[k] = fe_values[velocities].value(k, q);
                            phi_p[k] = fe_values[pressure].value(k, q);
                        }

                        for (const unsigned int i : fe_face_values.dof_indices()) {
                            for (const unsigned int j : fe_face_values.dof_indices()) {

                                local_matrix(i, j) +=
                                        (-(grad_phi_u[i] * normal) * phi_u[j]  // -(n * grad u, v)
                                         - (grad_phi_u[j] * normal) * phi_u[i] // -(n * grad v, u)
                                         + mu * (phi_u[i] * phi_u[j])          // mu (u, v)
                                         + (normal * phi_u[j]) * phi_p[i]      // (n * v, p)
                                         + (normal * phi_u[i]) * phi_p[j]      // (n * u, q)
                                        ) * fe_face_values.JxW(q);             // dx
                            }

                            Tensor<1, dim> prod_r = mu * phi_u[i] - grad_phi_u[i] * normal + phi_p[i] * normal;
                            // TODO må finnes en oneliner for Vector(dim) * Tensor<1, dim>...
                            // eller skal jeg gjøre som i step-22, er det en primitive?
                            // Calculate the inner product "manually" because one is a Vector and the other a Tensor.
                            double face_ips = 0;
                            for (unsigned int k = 0; k < dim; ++k) {
                                face_ips += bdd_values[q](k) * prod_r[k];  // (g, mu v - n grad v + q * n)
                            }

                            local_rhs(i) +=
                                    face_ips        //
                                    * fe_values.JxW(q);  // dx
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
                                      local_matrix(i, j));
                }
                system_rhs(local_dof_indices[i]) += local_rhs(i);
            }
        }
    }

    template<int dim>
    void StokesNitsche<dim>::solve() {
        // TODO annen løser? Løs på blokk-form?
        SparseDirectUMFPACK inverse;
        inverse.initialize(system_matrix);
        inverse.vmult(solution, system_rhs);
    }

    template<int dim>
    void StokesNitsche<dim>::output_results() const {
        // TODO se også Handling VVP.
        // see step-22
        std::vector<std::string> solution_names(dim, "velocity");
        solution_names.emplace_back("pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation> dci(
                dim, DataComponentInterpretation::component_is_part_of_vector);
        dci.push_back(DataComponentInterpretation::component_is_scalar);

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, solution_names, DataOut<dim>::type_dof_data, dci);

        data_out.build_patches();
        std::ofstream output("nitsche-stokes.vtk");
        data_out.write_vtk(output);
        std::cout << "  Output written to .vtk file." << std::endl;
    }

    template<int dim>
    void StokesNitsche<dim>::run() {
        make_grid();
        setup_dofs();
        assemble_system();
        solve();
        output_results();
        // TODO refinement
    }

} // namespace Stokes

int main() {
    std::cout << "StokesNitsche" << std::endl;
    {
        using namespace Stokes;
        StokesNitsche<2> stokesNitsche(1);
        stokesNitsche.run();
    }
}