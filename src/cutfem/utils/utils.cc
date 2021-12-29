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



    template
    class Selector<2>;

    template
    class Selector<3>;
}
