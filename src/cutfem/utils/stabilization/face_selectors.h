#ifndef INCLUDE_CUTFEM_STABILIZATION_FACE_SELECTORS_H_
#define INCLUDE_CUTFEM_STABILIZATION_FACE_SELECTORS_H_

#include <deal.II/grid/tria.h>

#include <deal.II/non_matching/mesh_classifier.h>

namespace cutfem
{
  namespace stabilization
  {
    using namespace dealii;
    using NonMatching::LocationToLevelSet;

    /**
     * An interface which for a given face of a cell defines whether
     * stabilization should be added to the face or not.
     */
    template <int dim>
    class FaceSelector
    {
    public:
      virtual ~FaceSelector() = default;

      /**
       * Returns true if the stabilization should be added to the face_index:th
       * face of the incoming cell.
       */
      virtual bool
      face_should_be_stabilized(
        const typename hp::DoFHandler<dim>::active_cell_iterator &cell,
        const unsigned int face_index) const = 0;
    };


    /**
     * Class which defined whether stabilization should be added based on the
     * LocationToInterface of two neighboring cells.
     *
     * The constructor takes a MeshClassifier and a function which given the
     * LocationToInterface of two incoming cells, returns whether stabilization
     * should be added or not.
     *
     * Typically the incoming function would be either
     * inside_stabilization or outside_stabilization.
     */
    template <int dim>
    class LocationBasedFaceSelector : public FaceSelector<dim>
    {
    public:
      /**
       * Constructor, takes the mesh classifier and the function describing
       * whether stabilization should be added or not.
       */
      LocationBasedFaceSelector(
        const NonMatching::MeshClassifier<dim> &mesh_classifier,
        const std::function<bool(const LocationToLevelSet cell1_position,
                                 const LocationToLevelSet cell2_position)>
          &face_between_should_be_stabilized);

      /**
       * Returns true if the stabilization should be added to the face_index:th
       * face of the incoming cell.
       */
      bool
      face_should_be_stabilized(
        const typename hp::DoFHandler<dim>::active_cell_iterator &cell,
        const unsigned int face_index) const override;

    private:
      const SmartPointer<const NonMatching::MeshClassifier<dim>>
        mesh_classifier;

      /**
       * Function which describes whether stabilization should be added to the
       * face between the two incoming cells.
       */
      const std::function<bool(const LocationToLevelSet cell1_position,
                               const LocationToLevelSet cell2_position)>
        face_between_should_be_stabilized;
    };


    /**
     * When we solve for a domain corresponding to the inside of a level set
     * function psi<0. Returns true if stabilization should be added to
     * the face between the cells based on the incoming LocationToInterface of
     * two neighboring cells
     */
    bool
    inside_stabilization(const LocationToLevelSet cell1_position,
                         const LocationToLevelSet cell2_position);

    /**
     * When we solve for a domain corresponding to the outside of a level set
     * function psi<0. Returns true if stabilization should be added to
     * the face between the cells based on the incoming LocationToInterface of
     * two neighboring cells
     */
    bool
    outside_stabilization(const NonMatching::LocationToLevelSet cell1_position,
                          const NonMatching::LocationToLevelSet cell2_position);

  } // namespace stabilization

} // namespace cutfem

#endif /* INCLUDE_CUTFEM_STABILIZATION_FACE_SELECTORS_H_ */
