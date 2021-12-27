#ifndef INCLUDE_CUTFEM_STABILIZATION_NORMAL_DERIVATIVE_COMPUTER_H_
#define INCLUDE_CUTFEM_STABILIZATION_NORMAL_DERIVATIVE_COMPUTER_H_

#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values.h>

#include <type_traits>

namespace cutfem
{
  namespace stabilization
  {
    using namespace dealii;

    template <int dim, class EXTRACTOR = FEValuesExtractors::Scalar>
    class NormalDerivativeComputer
    {
    public:
      typedef typename std::conditional<
        std::is_same<EXTRACTOR, FEValuesExtractors::Scalar>::value,
        FEValuesViews::Scalar<dim>,
        FEValuesViews::Vector<dim>>::type VIEWS;

      typedef typename VIEWS::value_type value_type;

      NormalDerivativeComputer(const FEFaceValues<dim> &fe_face_values,
                               const EXTRACTOR &        extractor);

      value_type
      normal_derivative(const unsigned int order,
                        const int          shape_index,
                        const int          quadrature_point);

      void
      use_reversed_face_normals();

    private:
      const SmartPointer<const FEFaceValues<dim>> fe_face_values;
      const EXTRACTOR                             extractor;
      double                                      normal_orientation = 1;
    };

  } /* namespace stabilization */
} // namespace cutfem

#endif /* INCLUDE_CUTFEM_STABILIZATION_NORMAL_DERIVATIVE_COMPUTER_H_ */
