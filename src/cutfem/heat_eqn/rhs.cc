#include "rhs.h"


#define pi 3.141592653589793

namespace examples::cut::HeatEquation {


    template<int dim>
    RightHandSide<dim>::RightHandSide(const double nu,
                                      const double tau,
                                      const double center_x,
                                      const double center_y)
            : nu(nu), tau(tau), center_x(center_x), center_y(center_y) {}

    template<int dim>
    double
    RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const {
        double x = p[0] - center_x;
        double y = p[1] - center_y;
        double t = this->get_time();
        return 2 * nu * pi * pi * exp(-t) * sin(pi * x) * sin(pi * y) -
               exp(-t) * sin(pi * x) * sin(pi * y);
    }


    template<int dim>
    BoundaryValues<dim>::BoundaryValues(const double center_x,
                                        const double center_y)
            : center_x(center_x), center_y(center_y) {}

    template<int dim>
    double
    BoundaryValues<dim>::value(const Point<dim> &p, const unsigned int) const {
        double x = p[0] - center_x;
        double y = p[1] - center_y;
        double t = this->get_time();
        return sin(pi * x) * sin(pi * y) * exp(-t);
    }


    template<int dim>
    AnalyticalSolution<dim>::AnalyticalSolution(const double center_x,
                                                const double center_y)
            : center_x(center_x), center_y(center_y) {}

    template<int dim>
    double AnalyticalSolution<dim>::
    value(const Point<dim> &p, const unsigned int component) const {
        (void) component;
        double x = p[0] - center_x;
        double y = p[1] - center_y;
        double t = this->get_time();
        return sin(pi * x) * sin(pi * y) * exp(-t);
    }

    template<int dim>
    Tensor<1, dim> AnalyticalSolution<dim>::
    gradient(const Point<dim> &p, const unsigned int component) const {
        (void) component;
        double x = p[0] - center_x;
        double y = p[1] - center_y;
        double t = this->get_time();
        Tensor<1, dim> value;
        value[0] = pi * sin(pi * y) * cos(pi * x) * exp(-t);
        value[1] = pi * sin(pi * x) * cos(pi * y) * exp(-t);

        return value;
    }


    template<int dim>
    FlowerDomain<dim>::FlowerDomain(const double center_x,
                                    const double center_y)
            : center_x(center_x), center_y(center_y) {}


    template<int dim>
    double FlowerDomain<dim>::
    value(const Point<dim> &p, const unsigned int component) const {
        // Uses the domain from eq (2.122) in Gürkan–Massing (2019), for z = 0.
        (void) component;
        double r = 0.5;
        double r0 = 3.5;

        double x = p[0] - center_x;
        double y = p[1] - center_y;
        return sqrt(pow(x, 2) + pow(y, 2)) - r +
               (r / r0) * cos(5 * atan2(y, x));
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
        double x0 = 0.9 * (half_length - sphere_radius) * sin(2 * pi * t);
        double y0 = 0;
        double x = p[0];
        double y = p[1];
        return sqrt(pow(x - x0, 2) + pow(y - y0, 2)) - sphere_radius;
    }


    template
    class RightHandSide<2>;

    template
    class RightHandSide<3>;

    template
    class BoundaryValues<2>;

    template
    class BoundaryValues<3>;

    template
    class AnalyticalSolution<2>;

    template
    class AnalyticalSolution<3>;

    template
    class FlowerDomain<2>;

    template
    class FlowerDomain<3>;

    template
    class MovingDomain<2>;

    template
    class MovingDomain<3>;

} // namespace examples::cut::HeatEquation
