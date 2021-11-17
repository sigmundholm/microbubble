#include <deal.II/base/geometric_utilities.h>

#include <boost/math/special_functions/sinc.hpp>

#include <iostream>

#include "rhs_stat.h"

#define pi 3.141592653589793


namespace examples::cut::NavierStokes::stationary {

    template<int dim>
    RightHandSide<dim>::RightHandSide(const double nu)
            : TensorFunction<1, dim>(), nu(nu) {}

    template<int dim>
    Tensor<1, dim> RightHandSide<dim>::
    value(const Point<dim> &p) const {
        // The analytical solution used (Ethier-Steinman, 1994), solves the
        // homogeneous Navier-Stokes equations.
        double x = p[0];
        double y = p[1];
        double t = this->get_time();
        Tensor<1, dim> val;
        val[0] = -2 * pi * pi * nu * sin(pi * y) * cos(pi * x);
        val[1] = 2 * pi * pi * nu * sin(pi * x) * cos(pi * y);
        return val;
    }


    template<int dim>
    ConvectionField<dim>::ConvectionField(const double nu)
            : TensorFunction<1, dim>(), nu(nu) {}

    template<int dim>
    Tensor<1, dim> ConvectionField<dim>::
    value(const Point<dim> &p) const {
        // The analytical solution used (Ethier-Steinman, 1994), solves the
        // homogeneous Navier-Stokes equations.
        double x = p[0];
        double y = p[1];
        double t = this->get_time();
        Tensor<1, dim> val;
        val[0] = -sin(pi * y) * cos(pi * x);
        val[1] = sin(pi * x) * cos(pi * y);
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

        // Analytical solution to the Navier-Stokes equations in 2D, see
        // Ethier-Steinman (1994).
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
        double t = this->get_time();

        Tensor<2, dim> value;
        value[0][0] = pi * sin(pi * x) * sin(pi * y);
        value[0][1] = -pi * cos(pi * x) * cos(pi * y);
        value[1][0] = pi * cos(pi * x) * cos(pi * y);
        value[1][1] = -pi * sin(pi * x) * sin(pi * y);
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
        // Analytical solution to the Navier-Stokes equations in 2D, see
        // Ethier-Steinman (1994).
        return -cos(2 * pi * x) / 4 - cos(2 * pi * y) / 4;
    }

    template<int dim>
    Tensor<1, dim> AnalyticalPressure<dim>::
    gradient(const Point<dim> &p, const unsigned int component) const {
        (void) component;
        double x = p[0];
        double y = p[1];
        double t = this->get_time();

        Tensor<1, dim> value;
        value[0] = pi * sin(2 * pi * x) / 2;
        value[1] = pi * sin(2 * pi * y) / 2;
        return value;
    }


    template
    class RightHandSide<2>;

    template
    class ConvectionField<2>;

    template
    class AnalyticalVelocity<2>;

    template
    class AnalyticalPressure<2>;

} // namespace examples::cut::NavierStokes::stationary
