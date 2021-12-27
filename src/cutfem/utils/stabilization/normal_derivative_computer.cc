#include "cutfem/stabilization/normal_derivative_computer.h"

namespace cutfem
{
  namespace stabilization
  {
    template <int dim, class EXTRACTOR>
    NormalDerivativeComputer<dim, EXTRACTOR>::NormalDerivativeComputer(
      const FEFaceValues<dim> &fe_face_values,
      const EXTRACTOR &        extractor)
      : fe_face_values(&fe_face_values)
      , extractor(extractor)
    {}

    template <int dim, class EXTRACTOR>
    typename NormalDerivativeComputer<dim, EXTRACTOR>::value_type
    NormalDerivativeComputer<dim, EXTRACTOR>::normal_derivative(
      const unsigned int order,
      const int          shape_index,
      const int          quadrature_point)
    {
      const VIEWS &        fe_values_views = (*fe_face_values)[extractor];
      const Tensor<1, dim> normal =
        normal_orientation * fe_face_values->normal_vector(quadrature_point);
      typename NormalDerivativeComputer<dim, EXTRACTOR>::value_type
        normal_derivative;
      normal_derivative = 0;
      // Compute the tensor contract with normal from right until we get a
      // scalar back.
      switch (order)
        {
          case 0:
            normal_derivative =
              fe_values_views.value(shape_index, quadrature_point);
            break;
          case 1:
            normal_derivative =
              fe_values_views.gradient(shape_index, quadrature_point) * normal;
            break;
          case 2:
            normal_derivative =
              (fe_values_views.hessian(shape_index, quadrature_point) *
               normal) *
              normal;
            break;
          case 3:
            normal_derivative =
              ((fe_values_views.third_derivative(shape_index,
                                                 quadrature_point) *
                normal) *
               normal) *
              normal;
            break;
          default:
            Assert(false, ExcNotImplemented());
        }
      return normal_derivative;
    }

    template <int dim, class EXTRACTOR>
    void
    NormalDerivativeComputer<dim, EXTRACTOR>::use_reversed_face_normals()
    {
      normal_orientation = -1;
    }

    template class NormalDerivativeComputer<1, FEValuesExtractors::Scalar>;
    template class NormalDerivativeComputer<2, FEValuesExtractors::Scalar>;
    template class NormalDerivativeComputer<3, FEValuesExtractors::Scalar>;
    template class NormalDerivativeComputer<1, FEValuesExtractors::Vector>;
    template class NormalDerivativeComputer<2, FEValuesExtractors::Vector>;
    template class NormalDerivativeComputer<3, FEValuesExtractors::Vector>;

  } /* namespace stabilization */
} /* namespace cutfem */
