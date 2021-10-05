#ifndef MICROBUBBLE_STOKES_BDF2_RHS_H
#define MICROBUBBLE_STOKES_BDF2_RHS_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

using namespace dealii;


namespace examples::cut::StokesEquation {


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


    template<int dim>
    class MovingDomain : public Function<dim> {
    public :
        MovingDomain(const double sphere_radius,
                     const double half_length,
                     const double radius);

        double
        value(const Point<dim> &p, const unsigned int component) const override;

    private:
        const double sphere_radius;
        const double half_length;
        const double radius;
    };

} // namespace examples::cut::StokesEquation


#endif // MICROBUBBLE_STOKES_BDF2_RHS_H
