#include "face_selectors.h"

namespace cutfem
{
  namespace stabilization
  {
    template <int dim>
    LocationBasedFaceSelector<dim>::LocationBasedFaceSelector(
      const NonMatching::CutMeshClassifier<dim> &mesh_classifier,
      const std::function<bool(const LocationToLevelSet cell1_position,
                               const LocationToLevelSet cell2_position)>
        &face_between_should_be_stabilized)
      : mesh_classifier(&mesh_classifier)
      , face_between_should_be_stabilized(face_between_should_be_stabilized)
    {}



    template <int dim>
    bool
    cutfem::stabilization::LocationBasedFaceSelector<dim>::
      face_should_be_stabilized(
        const typename hp::DoFHandler<dim>::active_cell_iterator &cell,
        const unsigned int face_index) const
    {
      typename Triangulation<dim>::cell_iterator neighbor =
        cell->neighbor(face_index);

      const NonMatching::LocationToLevelSet cell_category =
        mesh_classifier->location_to_level_set(cell);
      const NonMatching::LocationToLevelSet neighbor_category =
        mesh_classifier->location_to_level_set(neighbor);

      return face_between_should_be_stabilized(cell_category,
                                               neighbor_category);
    }



    bool
    inside_stabilization(const LocationToLevelSet cell1_position,
                         const LocationToLevelSet cell2_position)
    {
      bool one_is_inside_or_intersected =
        (cell1_position == LocationToLevelSet::INTERSECTED) ||
        (cell1_position == LocationToLevelSet::INSIDE);
      bool two_is_inside_or_intersected =
        (cell2_position == LocationToLevelSet::INTERSECTED) ||
        (cell2_position == LocationToLevelSet::INSIDE);
      bool both_are_inside = (cell1_position == LocationToLevelSet::INSIDE) &&
                             (cell2_position == LocationToLevelSet::INSIDE);
      bool should_stabilize =
        (one_is_inside_or_intersected && two_is_inside_or_intersected) &&
        !both_are_inside;
      return should_stabilize;
    }



    bool
    outside_stabilization(const NonMatching::LocationToLevelSet cell1_position,
                          const NonMatching::LocationToLevelSet cell2_position)
    {
      bool one_is_inside_or_intersected =
        (cell1_position == LocationToLevelSet::INTERSECTED) ||
        (cell1_position == LocationToLevelSet::OUTSIDE);
      bool two_is_inside_or_intersected =
        (cell2_position == LocationToLevelSet::INTERSECTED) ||
        (cell2_position == LocationToLevelSet::OUTSIDE);
      bool both_are_outside = (cell1_position == LocationToLevelSet::OUTSIDE) &&
                              (cell2_position == LocationToLevelSet::OUTSIDE);
      bool should_stabilize =
        (one_is_inside_or_intersected && two_is_inside_or_intersected) &&
        !both_are_outside;
      return should_stabilize;
    }



    template class LocationBasedFaceSelector<1>;
    template class LocationBasedFaceSelector<2>;
    template class LocationBasedFaceSelector<3>;
  } // namespace stabilization
} // namespace cutfem
