#ifndef MICROBUBBLE_RHS_STAT_H
#define MICROBUBBLE_RHS_STAT_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

// #include "../utils/cutfem_problem.h"

using namespace dealii;


namespace examples::cut::NavierStokes::stationary {

    // using namespace utils::problems;


    template<int dim>
    class RightHandSide : public TensorFunction<1, dim> {
    public:
        RightHandSide(double nu);

        Tensor<1, dim>
        value(const Point<dim> &p) const override;

        const double nu;
    };


    template<int dim>
    class ConvectionField : public TensorFunction<1, dim> {
    public:
        ConvectionField(double nu);

        Tensor<1, dim>
        value(const Point<dim> &p) const override;

        const double nu;
    };


    template<int dim>
    class AnalyticalVelocity : public TensorFunction<1, dim> {
    public:
        AnalyticalVelocity(double nu);

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
} // namespace examples::cut::NavierStokes::stationary

#endif //MICROBUBBLE_RHS_STAT_H
