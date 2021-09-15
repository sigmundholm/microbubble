#ifndef MICROBUBBLE_PROJECTIONS_RHS_H
#define MICROBUBBLE_PROJECTIONS_RHS_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

using namespace dealii;

namespace examples::cut::projections {

    struct Error {
        double mesh_size = 0;
        double l2_error_u = 0;
        double h1_error_u = 0;
        double h1_semi_u = 0;
        double l2_error_p = 0;
        double h1_error_p = 0;
        double h1_semi_p = 0;
    };

    template<int dim>
    class RightHandSide : public TensorFunction<1, dim> {
    public:
        RightHandSide(const double delta, const double nu, const double tau);

        Tensor<1, dim>
        value(const Point<dim> &p) const override;

        const double delta;
        const double nu;
        const double tau;
    };


    template<int dim>
    class BoundaryValues : public TensorFunction<1, dim> {
    public:
        Tensor<1, dim>
        value(const Point<dim> &p) const override;
    };


    template<int dim>
    class AnalyticalVelocity : public TensorFunction<1, dim> {
    public:
        Tensor<1, dim>
        value(const Point<dim> &p) const override;

        Tensor<2, dim>
        gradient(const Point<dim> &p) const override;
    };


    template<int dim>
    class AnalyticalPressure : public Function<dim> {
    public:
        double
        value(const Point<dim> &p, const unsigned int component) const override;

        Tensor<1, dim>
        gradient(const Point<dim> &p,
                 const unsigned int component) const override;
    };

} // namespace examples::cut::projections


#endif // MICROBUBBLE_PROJECTIONS_RHS_H
