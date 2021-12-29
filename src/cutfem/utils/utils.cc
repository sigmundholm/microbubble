#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparsity_tools.h>

#include "utils.h"


namespace utils {

    using NonMatching::LocationToLevelSet;

    template<int dim>
    Selector<dim>::Selector(const NonMatching::MeshClassifier<dim> &mesh_classifier)
            : mesh_classifier(&mesh_classifier) {}

    template<int dim>
    bool Selector<dim>::
    face_should_be_stabilized(
            const typename hp::DoFHandler<dim>::active_cell_iterator &cell,
            const unsigned int face_index) const {
        if (cell->at_boundary(face_index))
            return false;

        // We shouldn't stabilize between FENothing elements.
        if (cell->get_fe().dofs_per_cell == 0)
            return false;

        if (cell->neighbor(face_index)->get_fe().dofs_per_cell == 0)
            return false;

        const LocationToLevelSet cell_location =
                mesh_classifier->location_to_level_set(cell);
        const LocationToLevelSet neighbor_location =
                mesh_classifier->location_to_level_set(
                        cell->neighbor(face_index));

        // If both elements are inside we should't add stabilization
        if (cell_location == LocationToLevelSet::inside &&
            neighbor_location == LocationToLevelSet::inside)
            return false;

        return true;
    }


    template<int dim>
    Tools<dim>::Tools() {}

    template<int dim>
    void Tools<dim>::
    project(DoFHandler<dim> &dof_handler, 
                 AffineConstraints<double> &constraints,
                 FE_Q<dim> &fe,
                 QGauss<dim> quadrature_formula, 
                 Function<dim> &function, 
                 LA::MPI::Vector &solution) {

        MPI_Comm mpi_communicator = solution.get_mpi_communicator();

        // initiallize matrices
        IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler, 
                                                locally_relevant_dofs);
        

        LA::MPI::SparseMatrix system_matrix;
        LA::MPI::Vector system_rhs;

        system_rhs.reinit(locally_owned_dofs, mpi_communicator);
        DynamicSparsityPattern dsp(locally_relevant_dofs);
        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
        SparsityTools::distribute_sparsity_pattern(dsp, 
                                                   locally_owned_dofs,
                                                   mpi_communicator,
                                                   locally_relevant_dofs);
        system_matrix.reinit(locally_owned_dofs,
                                locally_owned_dofs, 
                                dsp,
                                mpi_communicator); 

        // Assemble
        // -------------------------------------------------------------------
        FEValues<dim> fe_v(fe,
                                quadrature_formula,
                                update_values | update_quadrature_points | update_JxW_values);
        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::vector<double> rhs_values(n_q_points);

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            if (cell->is_locally_owned()) {
                cell_matrix = 0.;
                cell_rhs = 0.;
                
                fe_v.reinit(cell);
                function.value_list(fe_v.get_quadrature_points(), rhs_values);

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
                    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                            cell_matrix(i, j) += fe_v.shape_value(i, q_point) *
                                fe_v.shape_value(j, q_point) *
                                fe_v.JxW(q_point);
                        }
                        cell_rhs(i) += rhs_values[q_point] *
                            fe_v.shape_value(i, q_point) *
                            fe_v.JxW(q_point);
                    }
                }
                cell->get_dof_indices(local_dof_indices);
                constraints.distribute_local_to_global(cell_matrix,
                                                    cell_rhs,
                                                    local_dof_indices,
                                                    system_matrix,
                                                    system_rhs);
            }
        }
        system_matrix.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);


        // Solve
        // --------------------------------------------------------------------
        SolverControl cn;
        PETScWrappers::SparseDirectMUMPS solver(cn, mpi_communicator);
        solver.set_symmetric_mode(false);
        LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                        mpi_communicator);
        solver.solve(system_matrix, completely_distributed_solution, system_rhs);
        constraints.distribute(completely_distributed_solution);
        
        solution = completely_distributed_solution;
    }


    template
    class Selector<2>;

    template
    class Selector<3>;

    template 
    class Tools<2>;
    
    template 
    class Tools<3>;
    
}
