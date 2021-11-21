#ifndef MICROBUBBLE_NAVIER_STOKES_RHS_H
#define MICROBUBBLE_NAVIER_STOKES_RHS_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include "../utils/cutfem_problem.h"

using namespace dealii;


namespace examples::cut::NavierStokes {

    using namespace utils::problems;


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
    class BoundaryValues : public TensorFunction<1, dim> {
    public:
        BoundaryValues(double nu);

        Tensor<1, dim>
        value(const Point<dim> &p) const override;

        const double nu;
    };


    template<int dim>
    class ParabolicFlow : public TensorFunction<1, dim> {
    public:
        ParabolicFlow(double radius, double max_speed, const double half_length,
                      bool stationary);

        Tensor<1, dim>
        value(const Point<dim> &p) const override;

        double
        get_current_max_speed();

        const double radius;
        const double half_length;
        const double max_speed;
        const bool stationary;
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


    template<int dim>
    class MovingDomain : public LevelSet<dim> {
    public :
        MovingDomain(double sphere_radius,
                     double half_length,
                     double radius);

        double
        value(const Point<dim> &p, unsigned int component) const override;

        Tensor<1, dim>
        get_velocity() override;

    private:
        const double sphere_radius;
        const double half_length;
        const double radius;
    };


    template<int dim>
    class Sphere : public LevelSet<dim> {
    public :
        Sphere(double sphere_radius,
               double center_x,
               double center_y);

        double
        value(const Point<dim> &p, unsigned int component) const override;

        double get_radius();

        Point<dim> get_center();

    private:
        const double sphere_radius;
        const double center_x;
        const double center_y;
    };

} // namespace examples::cut::NavierStokes

#endif // MICROBUBBLE_NAVIER_STOKES_RHS_H
