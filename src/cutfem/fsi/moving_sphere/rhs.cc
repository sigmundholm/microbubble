#include <cmath>

#include "rhs.h"


namespace cut::fsi::moving_sphere {

    template<int dim>
    BoundaryValues<dim>::BoundaryValues(const double half_length,
                                        const double radius,
                                        const double sphere_radius,
                                        LevelSet<dim> &domain)
            : TensorFunction<1, dim>(), half_length(half_length),
              radius(radius), sphere_radius(sphere_radius) {
        level_set = &domain;
    }

    template<int dim>
    Tensor<1, dim> BoundaryValues<dim>::
    value(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];

        Tensor<1, dim> zero_velocity;
        if (x == -half_length || x == half_length
            || y == -radius || y == radius) {
            // Zero Dirichlet boundary conditions on the whole boundary.
            return zero_velocity;
        }
        Tensor<1, dim> vel = level_set->get_velocity();
        return vel;
    }


    template<int dim>
    MovingDomain<dim>::MovingDomain(const double half_length,
                                    const double radius,
                                    const double sphere_radius,
                                    const double y_coord)
            :  half_length(half_length), radius(radius),
               sphere_radius(sphere_radius), y_coord(y_coord) {}

    template<int dim>
    double MovingDomain<dim>::
    value(const Point<dim> &p, const unsigned int component) const {
        (void) component;
        double t = this->get_time();
        double x0 = -0.9 * (half_length - sphere_radius) * cos(M_PI * t / 8);
        double y0 = y_coord;
        double x = p[0];
        double y = p[1];
        return -sqrt(pow(x - x0, 2) + pow(y - y0, 2)) + sphere_radius;
    }

    template<int dim>
    Tensor<1, dim> MovingDomain<dim>::
    get_velocity() {
        double t = this->get_time();
        Tensor<1, dim> v;
        v[0] = M_PI / 8 * 0.9 * (half_length - sphere_radius) *
               sin(M_PI * t / 8);
        return v;
    }


    template
    class BoundaryValues<2>;

    template
    class MovingDomain<2>;
}
