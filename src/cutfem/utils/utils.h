#ifndef MICROBUBBLE_UTILS_UTILS_H
#define MICROBUBBLE_UTILS_UTILS_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <deal.II/lac/generic_linear_algebra.h>


namespace LA 
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
!(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))

using namespace dealii::LinearAlgebraPETSc;
# define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
using namespace dealii::LinearAlgebraTrilinos;
#else
# error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/non_matching/mesh_classifier.h>
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
        Selector(const NonMatching::MeshClassifier<dim> &mesh_classifier);

        bool
        face_should_be_stabilized(
                const typename hp::DoFHandler<dim>::active_cell_iterator &cell,
                const unsigned int face_index) const override;

    private:
        const SmartPointer<const NonMatching::MeshClassifier<dim>> mesh_classifier;
    };
}

#endif //MICROBUBBLE_UTILS_UTILS_H
