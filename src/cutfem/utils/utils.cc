#include <deal.II/hp/dof_handler.h>

#include "utils.h"


namespace utils {

    using NonMatching::LocationToLevelSet;

    template<int dim>
    Selector<dim>::Selector(const NonMatching::CutMeshClassifier<dim> &mesh_classifier)
            : mesh_classifier(&mesh_classifier) {}

    // TODO are we here stabilizing all faces, and not just the
    //  intersected ones (plus outside ones)? See Guerkkan-Massing eq: (2.91) and prop. 2.21.
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
        if (cell_location == LocationToLevelSet::INSIDE &&
            neighbor_location == LocationToLevelSet::INSIDE)
            return false;

        return true;
    }


    template
    class Selector<2>;

    template
    class Selector<3>;
}
