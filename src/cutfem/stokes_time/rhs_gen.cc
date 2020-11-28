#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometric_utilities.h>

#include <boost/math/special_functions/sinc.hpp>

#include <iostream>

#include "rhs_gen.h"

#define pi 3.141592653589793


namespace TimeDependentStokesIE {


    template<int dim>
    RightHandSide<dim>::RightHandSide(const double nu,
                                      const double tau)
            : TensorFunction<1, dim>(), nu(nu), tau(tau) {}

    template<int dim>
    Tensor<1, dim> RightHandSide<dim>::
    value(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];
        double t = this->get_time();

        Tensor<1, dim> val;
        val[0] = -delta * exp(-2 * pi * pi * nu * t) * sin(pi * y) *
                 cos(pi * x) -
                 2 * pi * pi * nu * tau * exp(-2 * pi * pi * nu * t) *
                 sin(pi * y) * cos(pi * x) +
                 pi * tau * exp(-4 * pi * pi * nu * t) * sin(2 * pi * x) / 2;
        val[1] =
                delta * exp(-2 * pi * pi * nu * t) * sin(pi * x) * cos(pi * y) +
                2 * pi * pi * nu * tau * exp(-2 * pi * pi * nu * t) *
                sin(pi * x) * cos(pi * y) +
                pi * tau * exp(-4 * pi * pi * nu * t) * sin(2 * pi * y) / 2;
        return val;
    }


    template<int dim>
    BoundaryValues<dim>::BoundaryValues(const double nu)
            : TensorFunction<1, dim>(), nu(nu) {}

    template<int dim>
    Tensor<1, dim> BoundaryValues<dim>::
    value(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];
        double t = this->get_time();

        Tensor<1, dim> val;
        val[0] = -exp(-2 * pi * pi * nu * t) * sin(pi * y) * cos(pi * x);
        val[1] = exp(-2 * pi * pi * nu * t) * sin(pi * x) * cos(pi * y);
        return val;
    }


    template<int dim>
    AnalyticalVelocity<dim>::AnalyticalVelocity(const double nu)
            : TensorFunction<1, dim>(), nu(nu) {}

    template<int dim>
    Tensor<1, dim> AnalyticalVelocity<dim>::
    value(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];
        double t = this->get_time();

        Tensor<1, dim> val;
        val[0] = -exp(-2 * pi * pi * nu * t) * sin(pi * y) * cos(pi * x);
        val[1] = exp(-2 * pi * pi * nu * t) * sin(pi * x) * cos(pi * y);
        return val;
    }

    template<int dim>
    Tensor<2, dim> AnalyticalVelocity<dim>::
    gradient(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];
        double t = this->get_time();

        Tensor<2, dim> value;
        value[0][0] = pi * exp(-2 * pi * pi * nu * t) *
                      sin(pi * x) * sin(pi * y);
        value[0][1] = -pi * exp(-2 * pi * pi * nu * t) *
                      cos(pi * x) * cos(pi * y);
        value[1][0] = pi * exp(-2 * pi * pi * nu * t) *
                      cos(pi * x) * cos(pi * y);
        value[1][1] = -pi * exp(-2 * pi * pi * nu * t) *
                      sin(pi * x) * sin(pi * y);
        return value;
    }


    template<int dim>
    AnalyticalPressure<dim>::AnalyticalPressure(const double nu)
            : Function<dim>(1), nu(nu) {}

    template<int dim>
    double AnalyticalPressure<dim>::
    value(const Point<dim> &p, const unsigned int component) const {
        (void) component;
        double x = p[0];
        double y = p[1];
        double t = this->get_time();

        return (-cos(2 * pi * x) / 4 - cos(2 * pi * y) / 4) *
               exp(-4 * pi * pi * nu * t);
    }

    template<int dim>
    Tensor<1, dim> AnalyticalPressure<dim>::
    gradient(const Point<dim> &p, const unsigned int component) const {
        (void) component;
        double x = p[0];
        double y = p[1];
        double t = this->get_time();

        Tensor<1, dim> value;
        value[0] = pi * exp(-4 * pi * pi * nu * t) * sin(2 * pi * x) / 2;
        value[1] = pi * exp(-4 * pi * pi * nu * t) * sin(2 * pi * y) / 2;
        return value;
    }

    template
    class RightHandSide<2>;

    template
    class BoundaryValues<2>;

    template
    class AnalyticalVelocity<2>;

    template
    class AnalyticalPressure<2>;

} // namespace GeneralizedStokes