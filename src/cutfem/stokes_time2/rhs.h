#ifndef MICROBUBBLE_STOKES_BDF2_RHS_H
#define MICROBUBBLE_STOKES_BDF2_RHS_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

using namespace dealii;

namespace TimeDependentStokesBDF2 {

    struct Error {
        double mesh_size = 0;
        double time_step = 0;
        double l2_error_u = 0;
        double h1_error_u = 0;
        double h1_semi_u = 0;
        double l2_error_p = 0;
        double h1_error_p = 0;
        double h1_semi_p = 0;

        double l_inf_l2_error_u = 0;
        double l_inf_h1_error_u = 0;
    };

    template<int dim>
    class RightHandSide : public TensorFunction<1, dim> {
    public:
        RightHandSide(const double nu);

        Tensor<1, dim>
        value(const Point<dim> &p) const override;

        const double nu;
    };


    template<int dim>
    class BoundaryValues : public TensorFunction<1, dim> {
    public:
        BoundaryValues(const double nu);

        Tensor<1, dim>
        value(const Point<dim> &p) const override;

        const double nu;
    };


    template<int dim>
    class AnalyticalVelocity : public TensorFunction<1, dim> {
    public:
        AnalyticalVelocity(const double nu);

        Tensor<1, dim>
        value(const Point<dim> &p) const override;

        Tensor<2, dim>
        gradient(const Point<dim> &p) const override;

        const double nu;
    };


    template<int dim>
    class AnalyticalPressure : public Function<dim> {
    public:
        AnalyticalPressure(const double nu);

        double
        value(const Point<dim> &p, const unsigned int component) const override;

        Tensor<1, dim>
        gradient(const Point<dim> &p,
                 const unsigned int component) const override;

        const double nu;
    };

} // namespace TimeDependentStokesBDF2


#endif // MICROBUBBLE_STOKES_BDF2_RHS_H
