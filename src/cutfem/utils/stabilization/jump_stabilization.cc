#include "jump_stabilization.h"

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include "fe_collection_properties.h"
#include "jump_stabilization.templates.h"
#include "normal_derivative_computer.h"

namespace cutfem
{
  namespace stabilization
  {
    template <int dim, class EXTRACTOR>
    SingleFaceJumpStabilization<dim, EXTRACTOR>::SingleFaceJumpStabilization(
      const FEFaceValues<dim> &fe_values_cell,
      const FEFaceValues<dim> &fe_values_neighbor,
      const EXTRACTOR &        extractor,
      const WeightFunction &   weight_function)
      : fe_values_cell(&fe_values_cell)
      , fe_values_neighbor(&fe_values_neighbor)
      , extractor(extractor)
      , jump_weight(weight_function)
    {
      highest_element_order =
        std::max(fe_values_cell.get_fe().tensor_degree(),
                 fe_values_neighbor.get_fe().tensor_degree());
    }

    template <int dim, class EXTRACTOR>
    void
    SingleFaceJumpStabilization<dim, EXTRACTOR>::compute_stabilization()
    {
      save_n_dofs_for_cell_and_neighbor();
      setup_local_matrices();
      compute_local_matrices();
    }


    template <int dim, class EXTRACTOR>
    void
    SingleFaceJumpStabilization<dim, EXTRACTOR>::compute_h_scalings()
    {
      const typename Triangulation<dim>::active_cell_iterator cell =
        fe_values_cell->get_cell();

      const double h = std::pow(cell->measure(), 1. / dim);

      h_power_2i_plus_1.resize(highest_element_order + 1);
      for (unsigned int i = 0; i <= highest_element_order; ++i)
        h_power_2i_plus_1[i] = std::pow(h, 2. * i + 1.);
    }

    template <int dim, class EXTRACTOR>
    void
    SingleFaceJumpStabilization<dim, EXTRACTOR>::compute_local_matrices()
    {
      compute_h_scalings();

      const unsigned int n_quadrature_points =
        fe_values_cell->n_quadrature_points;

      std::vector<derivative_type> ddns_cell;
      std::vector<derivative_type> ddns_neighbor;
      for (unsigned int order = 0; order <= highest_element_order; ++order)
        {
          const double weight    = jump_weight(order, highest_element_order);
          const double h_scaling = h_power_2i_plus_1[order];

          for (unsigned int q = 0; q < n_quadrature_points; ++q)
            {
              compute_normal_derivatives(
                *fe_values_cell, order, q, false, ddns_cell);
              compute_normal_derivatives(
                *fe_values_neighbor, order, q, true, ddns_neighbor);
              const double JxW = fe_values_cell->JxW(q);

              for (unsigned int i = 0; i < n_dofs_cell; ++i)
                {
                  for (unsigned int j = 0; j < n_dofs_cell; ++j)
                    {
                      stab_cell_cell(i, j) +=
                        weight * h_scaling * ddns_cell[i] * ddns_cell[j] * JxW;
                    }
                  for (unsigned int j = 0; j < n_dofs_neighbor; ++j)
                    {
                      stab_cell_neighbor(i, j) += -weight * h_scaling *
                                                  ddns_cell[i] *
                                                  ddns_neighbor[j] * JxW;
                    }
                }

              for (unsigned int i = 0; i < n_dofs_neighbor; ++i)
                {
                  for (unsigned int j = 0; j < n_dofs_cell; ++j)
                    {
                      stab_neighbor_cell(i, j) += -weight * h_scaling *
                                                  ddns_neighbor[i] *
                                                  ddns_cell[j] * JxW;
                    }
                  for (unsigned int j = 0; j < n_dofs_neighbor; ++j)
                    {
                      stab_neighbor_neighbor(i, j) += weight * h_scaling *
                                                      ddns_neighbor[i] *
                                                      ddns_neighbor[j] * JxW;
                    }
                }
            }
        }
    }

    template <int dim, class EXTRACTOR>
    void
    SingleFaceJumpStabilization<dim, EXTRACTOR>::setup_local_matrices()
    {
      stab_cell_cell     = FullMatrix<double>(n_dofs_cell, n_dofs_cell);
      stab_cell_neighbor = FullMatrix<double>(n_dofs_cell, n_dofs_neighbor);
      stab_neighbor_cell = FullMatrix<double>(n_dofs_neighbor, n_dofs_cell);
      stab_neighbor_neighbor =
        FullMatrix<double>(n_dofs_neighbor, n_dofs_neighbor);
    }

    template <int dim, class EXTRACTOR>
    void
    SingleFaceJumpStabilization<dim,
                                EXTRACTOR>::save_n_dofs_for_cell_and_neighbor()
    {
      n_dofs_cell     = fe_values_cell->get_fe().n_dofs_per_cell();
      n_dofs_neighbor = fe_values_neighbor->get_fe().n_dofs_per_cell();
    }

    template <int dim, class EXTRACTOR>
    void
    SingleFaceJumpStabilization<dim, EXTRACTOR>::compute_normal_derivatives(
      const FEFaceValues<dim> &     fe_values,
      const unsigned int            derivative_order,
      const unsigned int            quadrature_point,
      const bool                    reverse_normals,
      std::vector<derivative_type> &normal_derivatives)
    {
      NormalDerivativeComputer<dim, EXTRACTOR> derivative_computer(fe_values,
                                                                   extractor);
      if (reverse_normals)
        derivative_computer.use_reversed_face_normals();

      normal_derivatives.resize(fe_values.dofs_per_cell);

      for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
        normal_derivatives[i] =
          derivative_computer.normal_derivative(derivative_order,
                                                i,
                                                quadrature_point);
    }

    template <int dim, class EXTRACTOR>
    const FullMatrix<double> &
    SingleFaceJumpStabilization<dim, EXTRACTOR>::get_local_stab11() const
    {
      return stab_cell_cell;
    }

    template <int dim, class EXTRACTOR>
    const FullMatrix<double> &
    SingleFaceJumpStabilization<dim, EXTRACTOR>::get_local_stab12() const
    {
      return stab_cell_neighbor;
    }

    template <int dim, class EXTRACTOR>
    const FullMatrix<double> &
    SingleFaceJumpStabilization<dim, EXTRACTOR>::get_local_stab21() const
    {
      return stab_neighbor_cell;
    }

    template <int dim, class EXTRACTOR>
    const FullMatrix<double> &
    SingleFaceJumpStabilization<dim, EXTRACTOR>::get_local_stab22() const
    {
      return stab_neighbor_neighbor;
    }

    template class SingleFaceJumpStabilization<1, FEValuesExtractors::Scalar>;
    template class SingleFaceJumpStabilization<2, FEValuesExtractors::Scalar>;
    template class SingleFaceJumpStabilization<1, FEValuesExtractors::Vector>;
    template class SingleFaceJumpStabilization<2, FEValuesExtractors::Vector>;
    template class SingleFaceJumpStabilization<3, FEValuesExtractors::Vector>;
    template class SingleFaceJumpStabilization<3, FEValuesExtractors::Scalar>;

    template class JumpStabilization<1>;
    template void
    JumpStabilization<1>::add_stabilization_to_matrix(const double,
                                                      SparseMatrix<double> &);
    template class JumpStabilization<2>;
    template void
    JumpStabilization<2>::add_stabilization_to_matrix(const double,
                                                      SparseMatrix<double> &);
    template class JumpStabilization<3>;
    template void
    JumpStabilization<3>::add_stabilization_to_matrix(const double,
                                                      SparseMatrix<double> &);

    template class JumpStabilization<1, FEValuesExtractors::Vector>;
    template void
    JumpStabilization<1, FEValuesExtractors::Vector>::
      add_stabilization_to_matrix(const double, SparseMatrix<double> &);
    template class JumpStabilization<2, FEValuesExtractors::Vector>;
    template void
    JumpStabilization<2, FEValuesExtractors::Vector>::
      add_stabilization_to_matrix(const double, SparseMatrix<double> &);
    template class JumpStabilization<3, FEValuesExtractors::Vector>;
    template void
    JumpStabilization<3, FEValuesExtractors::Vector>::
      add_stabilization_to_matrix(const double, SparseMatrix<double> &);

#ifdef DEAL_II_WITH_PETSC

    // Template with PETScWrappers::SparseMatrix
    template void
    JumpStabilization<1>::add_stabilization_to_matrix(
      const double,
      PETScWrappers::SparseMatrix &);
    template void
    JumpStabilization<2>::add_stabilization_to_matrix(
      const double,
      PETScWrappers::SparseMatrix &);
    template void
    JumpStabilization<3>::add_stabilization_to_matrix(
      const double,
      PETScWrappers::SparseMatrix &);

    template void
    JumpStabilization<1, FEValuesExtractors::Vector>::
      add_stabilization_to_matrix(const double, PETScWrappers::SparseMatrix &);
    template void
    JumpStabilization<2, FEValuesExtractors::Vector>::
      add_stabilization_to_matrix(const double, PETScWrappers::SparseMatrix &);
    template void
    JumpStabilization<3, FEValuesExtractors::Vector>::
      add_stabilization_to_matrix(const double, PETScWrappers::SparseMatrix &);

    // PETScWrappers::MPI::SparseMatrix
    template void
    JumpStabilization<1>::add_stabilization_to_matrix(
      const double,
      PETScWrappers::MPI::SparseMatrix &);
    template void
    JumpStabilization<2>::add_stabilization_to_matrix(
      const double,
      PETScWrappers::MPI::SparseMatrix &);
    template void
    JumpStabilization<3>::add_stabilization_to_matrix(
      const double,
      PETScWrappers::MPI::SparseMatrix &);

    template void
    JumpStabilization<1, FEValuesExtractors::Vector>::
      add_stabilization_to_matrix(const double,
                                  PETScWrappers::MPI::SparseMatrix &);
    template void
    JumpStabilization<2, FEValuesExtractors::Vector>::
      add_stabilization_to_matrix(const double,
                                  PETScWrappers::MPI::SparseMatrix &);
    template void
    JumpStabilization<3, FEValuesExtractors::Vector>::
      add_stabilization_to_matrix(const double,
                                  PETScWrappers::MPI::SparseMatrix &);
#endif

#ifdef DEAL_II_WITH_TRILINOS
    template void
    JumpStabilization<1>::add_stabilization_to_matrix(
      const double,
      TrilinosWrappers::SparseMatrix &);
    template void
    JumpStabilization<2>::add_stabilization_to_matrix(
      const double,
      TrilinosWrappers::SparseMatrix &);
    template void
    JumpStabilization<3>::add_stabilization_to_matrix(
      const double,
      TrilinosWrappers::SparseMatrix &);

    template void
    JumpStabilization<1, FEValuesExtractors::Vector>::
      add_stabilization_to_matrix(const double,
                                  TrilinosWrappers::SparseMatrix &);
    template void
    JumpStabilization<2, FEValuesExtractors::Vector>::
      add_stabilization_to_matrix(const double,
                                  TrilinosWrappers::SparseMatrix &);
    template void
    JumpStabilization<3, FEValuesExtractors::Vector>::
      add_stabilization_to_matrix(const double,
                                  TrilinosWrappers::SparseMatrix &);
#endif


  } // namespace stabilization
} // namespace cutfem
