#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometric_utilities.h>

#include <boost/math/special_functions/sinc.hpp>

#include <iostream>

#include "rhs.h"

#define pi 3.141592653589793


namespace examples::cut::StokesEquation2 {


    template<int dim>
    RightHandSide<dim>::RightHandSide(const double delta,
                                      const double nu,
                                      const double tau)
            : TensorFunction<1, dim>(), delta(delta), nu(nu), tau(tau) {}

    template<int dim>
    Tensor<1, dim> RightHandSide<dim>::
    value(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];
        Tensor<1, dim> val;
        val[0] = -delta * sin(pi * y) * cos(pi * x) -
                 2 * pi * pi * nu * tau * sin(pi * y) * cos(pi * x) +
                 pi * tau * sin(2 * pi * x) / 2;
        val[1] = delta * sin(pi * x) * cos(pi * y) +
                 2 * pi * pi * nu * tau * sin(pi * x) * cos(pi * y) +
                 pi * tau * sin(2 * pi * y) / 2;
        return val;
    }

    template<int dim>
    Tensor<1, dim> BoundaryValues<dim>::
    value(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];
        Tensor<1, dim> val;
        val[0] = -sin(pi * y) * cos(pi * x);
        val[1] = sin(pi * x) * cos(pi * y);
        return val;
    }


    template<int dim>
    Tensor<1, dim> AnalyticalVelocity<dim>::
    value(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];
        Tensor<1, dim> val;
        val[0] = -sin(pi * y) * cos(pi * x);
        val[1] = sin(pi * x) * cos(pi * y);
        return val;
    }

    template<int dim>
    Tensor<2, dim> AnalyticalVelocity<dim>::
    gradient(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];
        Tensor<2, dim> value;
        value[0][0] = pi * sin(pi * x) * sin(pi * y);
        value[0][1] = -pi * cos(pi * x) * cos(pi * y);
        value[1][0] = pi * cos(pi * x) * cos(pi * y);
        value[1][1] = -pi * sin(pi * x) * sin(pi * y);
        return value;
    }

    template<int dim>
    double AnalyticalPressure<dim>::
    value(const Point<dim> &p, const unsigned int component) const {
        (void) component;
        double x = p[0];
        double y = p[1];
        return -cos(2 * pi * x) / 4 - cos(2 * pi * y) / 4;
    }

    template<int dim>
    Tensor<1, dim> AnalyticalPressure<dim>::
    gradient(const Point<dim> &p, const unsigned int component) const {
        (void) component;
        double x = p[0];
        double y = p[1];
        Tensor<1, dim> value;
        value[0] = pi * sin(2 * pi * x) / 2;
        value[1] = pi * sin(2 * pi * y) / 2;
        return value;
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
        double t = 0.5; //this->get_time();
        // Here it is assumed that T = 0.05, since for T = 1, the analytical
        // solution used is very small at the end time.
        double x0 = 0.9 * (half_length - sphere_radius) *
                    (2 * t/0.05 - 1); // sin(2 * pi * t);
        double y0 = 0.9 * (radius - sphere_radius) * (2 * t/0.05 - 1);
        double x = p[0];
        double y = p[1];
        return -sqrt(pow(x - x0, 2) + pow(y - y0, 2)) + sphere_radius;
    }

    template<int dim>
    Tensor<1, dim> MovingDomain<dim>::
    get_velocity() {
        Tensor<1, dim> val;
        val[0] = 0; // 0.9 * (half_length - sphere_radius) * 2 / 0.05;
        val[1] = 0; // 0.9 * (radius - sphere_radius) * 2 / 0.05;
        return val;
    }


    template
    class RightHandSide<2>;

    template
    class BoundaryValues<2>;

    template
    class AnalyticalVelocity<2>;

    template
    class AnalyticalPressure<2>;

    template
    class MovingDomain<2>;

} // namespace examples::cut::StokesEquation2