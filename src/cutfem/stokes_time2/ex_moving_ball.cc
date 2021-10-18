#include <boost/math/special_functions/sinc.hpp>

#include "ex_moving_ball.h"

#define pi 3.141592653589793


using namespace dealii;

namespace examples::cut::StokesEquation::ex1 {

    template<int dim>
    ZeroTensorFunction<dim>::ZeroTensorFunction() : TensorFunction<1, dim>() {}

    template<int dim>
    Tensor<1, dim> ZeroTensorFunction<dim>::
    value(const Point<dim> &p) const {
        Tensor<1, dim> val;
        val[0] = 0;
        val[1] = 0;
        return val;
    }

    template<int dim>
    Tensor<2, dim> ZeroTensorFunction<dim>::
    gradient(const Point<dim> &p) const {
        Tensor<2, dim> value;
        return value;
    }


    template<int dim>
    ZeroFunction<dim>::ZeroFunction() : Function<dim>() {}

    template<int dim>
    double ZeroFunction<dim>::
    value(const Point<dim> &p, const unsigned int component) const {
        return 0;
    }


    template<int dim>
    BoundaryValues<dim>::BoundaryValues(const double sphere_radius,
                                        const double half_length,
                                        const double radius)
            : TensorFunction<1, dim>(), sphere_radius(sphere_radius),
              half_length(half_length), radius(radius) {}

    template<int dim>
    Tensor<1, dim> BoundaryValues<dim>::
    value(const Point<dim> &p) const {
        double t = this->get_time();
        double x = p[0];
        double y = p[1];

        double x0;
        double y0;

        Tensor<1, dim> velocity;
        if (x == -half_length || x == half_length
            || y == -radius || y == radius) {
            // Zero Dirichlet boundary conditions on the whole boundary.
            return velocity;
        }
        x0 = -0.9 * (half_length - sphere_radius) * 2 * pi * sin(2 * pi * t);
        y0 = 0;
        velocity[0] = x0;
        velocity[1] = y0;
        return velocity;
    }


    template<int dim>
    MovingDomain<dim>::MovingDomain(const double sphere_radius,
                                    const double half_length,
                                    const double radius)
            : sphere_radius(sphere_radius), half_length(half_length),
              radius(radius) {}

    template<int dim>
    double MovingDomain<dim>::
    value(const Point<dim> &p, const unsigned int component) const {
        (void) component;
        double t = this->get_time();
        double x0 = 0.9 * (half_length - sphere_radius) * cos(2 * pi * t);
        double y0 = 0;
        double x = p[0];
        double y = p[1];
        return -sqrt(pow(x - x0, 2) + pow(y - y0, 2)) + sphere_radius;
    }

    template<int dim>
    Tensor<1, dim> MovingDomain<dim>::
    get_velocity() {
        double t = this->get_time();
        Tensor<1, dim> val;
        val[0] =  - 0.9 * (half_length - sphere_radius) * 2 * pi * sin(2 * pi * t);
        return val;
    }

    template
    class ZeroTensorFunction<2>;

    template
    class ZeroFunction<2>;

    template
    class BoundaryValues<2>;

    template
    class MovingDomain<2>;

}
