#ifndef INCLUDE_CUTFEM_STABILIZATION_JUMP_STABILIZATION_H_
#define INCLUDE_CUTFEM_STABILIZATION_JUMP_STABILIZATION_H_

#include <deal.II/base/smartpointer.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <boost/math/special_functions/factorials.hpp>

#include <cmath>
#include <memory>
#include <utility>

#include "fe_collection_properties.h"
#include "face_selectors.h"
#include "normal_derivative_computer.h"

namespace cutfem
{
  namespace stabilization
  {
    using namespace dealii;

    /**
     * Function defining the weights, $w_{k,p}$,
     * used in the jump-stabilization.
     * Here, k=derivative_order, p=element_order.
     *
     * \begin{equation}
     *   j_i(u,v)=\sum_{F} \sum_{k=1}^p
     *   w_{k,p} h^{2k+1} <[\partial_n^k u],[\partial_n^k v]>_F
     * \end{equation}
     */
    typedef std::function<double(const unsigned int derivative_order,
                                 const unsigned int element_order)>
      WeightFunction;

    /**
     * Weight-function describing the weights suggested by the Taylor-expansion.
     * That is, in the weight function above we have:
     *
     * \begin{equation}
     *   w_{k,p}=\frac{3}{(2k+1)(k!)^2}.
     * \end{equation}
     *
     * Note that the weights are scaled so that the $w_{1,p}=1$.
     */
    inline double
    taylor_weights(const unsigned int derivative_order,
                   const unsigned int /*element_order*/)
    {
      const double order_factorial =
        boost::math::factorial<double>(derivative_order);
      const double scaling_to_make_order_1_eq_1 = 3;
      const double weight =
        scaling_to_make_order_1_eq_1 /
        ((2. * derivative_order + 1.) * std::pow(order_factorial, 2.));
      return weight;
    }

    inline double
    unit_weights(const unsigned int /*derivative_order*/,
                 const unsigned int /*element_order*/)
    {
      return 1;
    }

    /**
     * The weights that we argue for in the Higher-Order wave equation
     * manuscript. We call these minimalizing here since we minimized a constant
     * in the paper. But they do not necessarily give a minimum in practice.
     */
    inline double
    minimalizing_weights(const unsigned int derivative_order,
                         const unsigned int element_order)
    {
      const double factorial = boost::math::factorial<double>(derivative_order);
      const double element_dependence =
        std::pow((double)element_order, 2.0 * derivative_order + 1.0);
      const double scaling_to_make_order_1_eq_1 = std::sqrt(3.);
      const double weight =
        scaling_to_make_order_1_eq_1 /
        (factorial * std::sqrt(2.0 * derivative_order + 1.0) *
         element_dependence);
      return weight;
    }

    /**
     * Computes the stabilization from a single face between two
     * cells. The stabilization contributes to all dofs of the
     * two cells. Because of this the contribution of this stabilization
     * is added to four local matrices.
     *
     * This class should only be used through JumpStabilization.
     */
    template <int dim, class EXTRACTOR = FEValuesExtractors::Scalar>
    class SingleFaceJumpStabilization
    {
    public:
      typedef typename NormalDerivativeComputer<dim, EXTRACTOR>::value_type
        derivative_type;

      SingleFaceJumpStabilization(){};

      SingleFaceJumpStabilization(const FEFaceValues<dim> &fe_values_cell,
                                  const FEFaceValues<dim> &fe_values_neighbor,
                                  const EXTRACTOR &        extractor,
                                  const WeightFunction &   jump_weight);

      void
      compute_stabilization();

      const FullMatrix<double> &
      get_local_stab11() const;

      const FullMatrix<double> &
      get_local_stab12() const;

      const FullMatrix<double> &
      get_local_stab21() const;

      const FullMatrix<double> &
      get_local_stab22() const;

    private:
      void
      save_n_dofs_for_cell_and_neighbor();

      void
      setup_local_matrices();

      /**
       * Pre-compute the h-scaling factor h^{2*i+1} (where i is the derivative
       * order) that appears in the stabilization, write these to the vector
       * h_power_2i_plus_1.
       */
      void
      compute_h_scalings();

      /**
       * Compute the normal derivative of order @param derivative_order
       * of all the shape functions of the incoming FEFaceValues object
       * at the quadrature point with index @param quadrature_point and
       * store these in the vector @param ddns.
       *
       * If @param reverse_normals is true the normal of the fe_values object
       * should be reversed.
       */
      void
      compute_normal_derivatives(
        const FEFaceValues<dim> &     fe_values,
        const unsigned int            derivative_order,
        const unsigned int            quadrature_point,
        const bool                    reverse_normals,
        std::vector<derivative_type> &normal_derivatives);

      void
      compute_local_matrices();

      /**
       * Vector containing precomputed values: h^{2*i+1}, where i is the vector
       * index.
       */
      std::vector<double> h_power_2i_plus_1;

      FullMatrix<double> stab_cell_cell, stab_cell_neighbor, stab_neighbor_cell,
        stab_neighbor_neighbor;

      const SmartPointer<const FEFaceValues<dim>> fe_values_cell,
        fe_values_neighbor;

      const EXTRACTOR extractor;

      WeightFunction jump_weight;

      unsigned int n_dofs_cell = 0, n_dofs_neighbor = 0,
                   highest_element_order = 1;
    };


    /**
     * This class computes a stabilization of the form.
     *  \begin{equation}
     *   j_i(u,v)=\sum_{F} \sum_{k=1}^p
     *   w_{k,p} h^{2k+1} <[\partial_n^k u],[\partial_n^k v]>_F
     * \end{equation}
     *
     * The use of this class is as follows. On each cell during assembling we
     * first call
     * jump_stabilization.compute_stabilization(cell);
     * to internally compute the stabilization. After that one adds
     * the stabilization to the desired matrix by the function
     * jump_stabilization.add_stabilization_to_matrix(scaling,matrix);
     */
    template <int dim,
              class EXTRACTOR = FEValuesExtractors::Scalar,
              class SINGLE_FACE_STABILIZATION =
                SingleFaceJumpStabilization<dim, EXTRACTOR>>
    class JumpStabilization
    {
    public:
      JumpStabilization(
        const DoFHandler<dim> &                    dof_handler,
        const hp::MappingCollection<dim> &         mapping_collection,
        const NonMatching::MeshClassifier<dim> &cut_mesh_classifier,
        const AffineConstraints<double> &          constraints);

      /**
       * Computes the stabilization that should be added to the faces of the
       * present cell, by looping over all the cell faces.
       */
      void
      compute_stabilization(
        const typename DoFHandler<dim>::active_cell_iterator &cell);

      /**
       * Add the computed stabilization scaled by "scaling"to the incoming
       * matrix.
       */
      template <class MATRIX>
      void
      add_stabilization_to_matrix(const double scaling, MATRIX &Matrix);

      /**
       * Returns Update flags so that all non-zero derivatives up to
       * max_element_order can be computed.
       */
      static UpdateFlags
      get_flags_up_to_right_order(const unsigned int max_element_order);

      /**
       * Set a function describing what faces to stabilize.
       * The incoming function should take in cell_positions of the two
       * cells sharing a common face and return a bool.
       */
      void
      set_function_describing_faces_to_stabilize(
        const std::function<bool(const NonMatching::LocationToLevelSet,
                                 const NonMatching::LocationToLevelSet)>
          &faceShouldBeStabilized);

      /**
       * Set an FaceSelector object, which describes which of the faces should
       * be stabilized.
       */
      void
      set_faces_to_stabilize(
        const std::shared_ptr<const FaceSelector<dim>> &face_selector);

      /**
       * For vector valued problems we can set an extractor describing
       * which field the stabilization should be added to.
       */
      void
      set_extractor(EXTRACTOR extractor);

      void
      set_weight_function(const WeightFunction &jump_weight);

    private:
      void
      setup_fe_face_values();

      template <class MATRIX>
      void
      add_stabilization_for_face(const unsigned int pair_index,
                                 const double       scaling,
                                 MATRIX &           Matrix);

      const SmartPointer<const DoFHandler<dim>>            dof_handler;
      const SmartPointer<const hp::MappingCollection<dim>> mapping_collection;

      const SmartPointer<const NonMatching::MeshClassifier<dim>>
        cut_mesh_classifier;

      EXTRACTOR extractor = EXTRACTOR(0);

      unsigned int current_cell_level = 0, current_cell_index = 0;

      const SmartPointer<const AffineConstraints<double>> constraints;

      WeightFunction jump_weight;

      std::shared_ptr<const FaceSelector<dim>> face_selector;

      std::unique_ptr<hp::FEFaceValues<dim>> fe_face_values_cell,
        fe_face_values_neighbor;

      std::vector<std::pair<int, SINGLE_FACE_STABILIZATION>>
        face_stabilizations;
    };

  } // namespace stabilization
} // namespace cutfem

#endif /* INCLUDE_CUTFEM_STABILIZATION_JUMP_STABILIZATION_H_ */
