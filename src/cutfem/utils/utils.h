#ifndef MICROBUBBLE_UTILS_UTILS_H
#define MICROBUBBLE_UTILS_UTILS_H

#include <deal.II/non_matching/cut_mesh_classifier.h>
#include <deal.II/lac/affine_constraints.h>

#include "stabilization/face_selectors.h"


using namespace dealii;
using namespace cutfem;

namespace utils {

    /*
    * Class defining which elements we should add stabilization.
    */
    template<int dim>
    class Selector : public stabilization::FaceSelector<dim> {
    public:
        Selector(const NonMatching::CutMeshClassifier<dim> &mesh_classifier);

        bool
        face_should_be_stabilized(
                const typename hp::DoFHandler<dim>::active_cell_iterator &cell,
                const unsigned int face_index) const override;

    private:
        const SmartPointer<const NonMatching::CutMeshClassifier<dim>> mesh_classifier;
    };
}

#endif //MICROBUBBLE_UTILS_UTILS_H
