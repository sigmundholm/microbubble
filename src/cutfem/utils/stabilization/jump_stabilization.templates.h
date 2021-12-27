/*
 * jump_stabilization.templates.h
 *
 *  Created on: Feb 22, 2017
 *      Author: simon
 */

#ifndef INCLUDE_CUTFEM_STABILIZATION_JUMP_STABILIZATION_TEMPLATES_H_
#define INCLUDE_CUTFEM_STABILIZATION_JUMP_STABILIZATION_TEMPLATES_H_

#include <deal.II/lac/sparse_matrix.h>

#include "jump_stabilization.h"

namespace cutfem
{
  namespace stabilization
  {
    using namespace dealii;

    template <int dim, class EXTRACTOR, class SINGLE_FACE_STABILIZATION>
    JumpStabilization<dim, EXTRACTOR, SINGLE_FACE_STABILIZATION>::
      JumpStabilization(
        const DoFHandler<dim> &                    dof_handler,
        const hp::MappingCollection<dim> &         mapping_collection,
        const NonMatching::CutMeshClassifier<dim> &cut_mesh_classifier,
        const AffineConstraints<double> &          constraints)
      : dof_handler(&dof_handler)
      , mapping_collection(&mapping_collection)
      , cut_mesh_classifier(&cut_mesh_classifier)
      , constraints(&constraints)
    {
      setup_fe_face_values();
      set_function_describing_faces_to_stabilize(inside_stabilization);
      jump_weight = taylor_weights;
    }

    template <int dim, class EXTRACTOR, class SINGLE_FACE_STABILIZATION>
    UpdateFlags
    JumpStabilization<dim, EXTRACTOR, SINGLE_FACE_STABILIZATION>::
      get_flags_up_to_right_order(const unsigned int max_element_order)
    {
      UpdateFlags update_flags = update_values | update_gradients |
                                 update_normal_vectors | update_JxW_values;
      switch (max_element_order)
        {
          case 3:
            update_flags =
              update_flags | update_3rd_derivatives | update_hessians;
            break;
          case 2:
            update_flags = update_flags | update_hessians;
        }
      return update_flags;
    }

    template <int dim, class EXTRACTOR, class SINGLE_FACE_STABILIZATION>
    void
    JumpStabilization<dim, EXTRACTOR, SINGLE_FACE_STABILIZATION>::
      setup_fe_face_values()
    {
      const hp::FECollection<dim> &fe_collection =
        dof_handler->get_fe_collection();
      const unsigned int max_element_order =
        cutfem::get_max_element_order(fe_collection);
      const UpdateFlags update_flags =
        get_flags_up_to_right_order(max_element_order);
      hp::QCollection<dim - 1> q_collection;
      q_collection.push_back(QGauss<dim - 1>(max_element_order + 1));
      fe_face_values_cell.reset(new hp::FEFaceValues<dim>(
        *mapping_collection, fe_collection, q_collection, update_flags));
      fe_face_values_neighbor.reset(new hp::FEFaceValues<dim>(
        *mapping_collection, fe_collection, q_collection, update_flags));
    }

    template <int dim, class EXTRACTOR, class SINGLE_FACE_STABILIZATION>
    void
    JumpStabilization<dim, EXTRACTOR, SINGLE_FACE_STABILIZATION>::
      compute_stabilization(
        const typename DoFHandler<dim>::active_cell_iterator &cell)
    {
      current_cell_index = cell->index();
      current_cell_level = cell->level();
      face_stabilizations.clear();
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          if (!(cell->at_boundary(f)))
            {
              typename hp::DoFHandler<dim>::cell_iterator neighbor =
                cell->neighbor(f);

              if (face_selector->face_should_be_stabilized(cell, f))
                {
                  fe_face_values_cell->reinit(cell, f);
                  fe_face_values_neighbor->reinit(
                    neighbor, cell->neighbor_of_neighbor(f));
                  face_stabilizations.push_back(
                    std::make_pair<int, SINGLE_FACE_STABILIZATION>(
                      f,
                      SINGLE_FACE_STABILIZATION(
                        fe_face_values_cell->get_present_fe_values(),
                        fe_face_values_neighbor->get_present_fe_values(),
                        extractor,
                        jump_weight)));
                  SINGLE_FACE_STABILIZATION &latest_stabilization =
                    face_stabilizations.back().second;
                  latest_stabilization.compute_stabilization();
                }
            }
        }
    }

    template <int dim, class EXTRACTOR, class SINGLE_FACE_STABILIZATION>
    template <class MATRIX>
    void
    JumpStabilization<dim, EXTRACTOR, SINGLE_FACE_STABILIZATION>::
      add_stabilization_to_matrix(const double scaling, MATRIX &Matrix)
    {
      typename hp::DoFHandler<dim>::active_cell_iterator cell(
        &(dof_handler->get_triangulation()),
        current_cell_level,
        current_cell_index,
        dof_handler);
      for (unsigned int i = 0; i < face_stabilizations.size(); ++i)
        {
          add_stabilization_for_face(i, scaling, Matrix);
        }
    }

    template <int dim, class EXTRACTOR, class SINGLE_FACE_STABILIZATION>
    template <class MATRIX>
    void
    JumpStabilization<dim, EXTRACTOR, SINGLE_FACE_STABILIZATION>::
      add_stabilization_for_face(const unsigned int pair_index,
                                 const double       scaling,
                                 MATRIX &           matrix)
    {
      const std::pair<unsigned int, SINGLE_FACE_STABILIZATION>
        &          index_stab_pair = face_stabilizations.at(pair_index);
      unsigned int face_index      = index_stab_pair.first;
      const SINGLE_FACE_STABILIZATION &face_stabilization =
        index_stab_pair.second;
      // Get iterators for cell and neighbor
      typename hp::DoFHandler<dim>::active_cell_iterator cell(
        &(dof_handler->get_triangulation()),
        current_cell_level,
        current_cell_index,
        dof_handler);
      typename hp::DoFHandler<dim>::active_cell_iterator neighbor =
        cell->neighbor(face_index);

      std::vector<types::global_dof_index> cell_dof_indices(
        cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(cell_dof_indices);
      std::vector<types::global_dof_index> neighbor_dof_indices(
        neighbor->get_fe().dofs_per_cell);
      neighbor->get_dof_indices(neighbor_dof_indices);

      FullMatrix<double> J11 = face_stabilization.get_local_stab11();
      J11 *= scaling;
      constraints->distribute_local_to_global(J11,
                                              cell_dof_indices,
                                              cell_dof_indices,
                                              matrix);

      FullMatrix<double> J12 = face_stabilization.get_local_stab12();
      J12 *= scaling;
      constraints->distribute_local_to_global(J12,
                                              cell_dof_indices,
                                              neighbor_dof_indices,
                                              matrix);

      FullMatrix<double> J21 = face_stabilization.get_local_stab21();
      J21 *= scaling;
      constraints->distribute_local_to_global(J21,
                                              neighbor_dof_indices,
                                              cell_dof_indices,
                                              matrix);

      FullMatrix<double> J22 = face_stabilization.get_local_stab22();
      J22 *= scaling;
      constraints->distribute_local_to_global(J22,
                                              neighbor_dof_indices,
                                              neighbor_dof_indices,
                                              matrix);
    }

    template <int dim, class EXTRACTOR, class SINGLE_FACE_STABILIZATION>
    void
    JumpStabilization<dim, EXTRACTOR, SINGLE_FACE_STABILIZATION>::
      set_function_describing_faces_to_stabilize(
        const std::function<bool(const NonMatching::LocationToLevelSet,
                                 const NonMatching::LocationToLevelSet)>
          &faceShouldBeStabilized)
    {
      face_selector.reset(
        new LocationBasedFaceSelector<dim>(*cut_mesh_classifier,
                                           faceShouldBeStabilized));
    }

    template <int dim, class EXTRACTOR, class SINGLE_FACE_STABILIZATION>
    void
    JumpStabilization<dim, EXTRACTOR, SINGLE_FACE_STABILIZATION>::
      set_faces_to_stabilize(
        const std::shared_ptr<const FaceSelector<dim>> &face_selector)
    {
      this->face_selector = face_selector;
    }

    template <int dim, class EXTRACTOR, class SINGLE_FACE_STABILIZATION>
    void
    JumpStabilization<dim, EXTRACTOR, SINGLE_FACE_STABILIZATION>::set_extractor(
      EXTRACTOR extractor)
    {
      this->extractor = extractor;
    }

    template <int dim, class EXTRACTOR, class SINGLE_FACE_STABILIZATION>
    void
    JumpStabilization<dim, EXTRACTOR, SINGLE_FACE_STABILIZATION>::
      set_weight_function(const WeightFunction &jump_weight)
    {
      this->jump_weight = jump_weight;
    }

  } // namespace stabilization
} // namespace cutfem

#endif /* INCLUDE_CUTFEM_STABILIZATION_JUMP_STABILIZATION_TEMPLATES_H_ */
