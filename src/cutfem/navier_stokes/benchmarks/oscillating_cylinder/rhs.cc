#include <cmath>

#include "rhs.h"


namespace examples::cut::NavierStokes::benchmarks::oscillating {

    template<int dim>
    BoundaryValues<dim>::BoundaryValues(const double half_length,
                                        const double radius,
                                        const double sphere_radius,
                                        LevelSet<dim> &domain)
            : ParabolicFlow<dim>(radius, 0, half_length, false) {
        level_set = &domain;
    }

    template<int dim>
    Tensor<1, dim> BoundaryValues<dim>::
    value(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];

        Tensor<1, dim> zero_velocity;
        if (x == -this->half_length || x == this->half_length
            || y == -this->radius || y == this->radius) {
            // Zero Dirichlet boundary conditions on the whole boundary.
            return zero_velocity;
        }
        Tensor<1, dim> vel = level_set->get_velocity();
        return vel;
    }


    template<int dim>
    OscillatingSphere<dim>::OscillatingSphere(const double sphere_radius, 
                                              const double center_x, 
                                              const double center_y, 
                                              const double amplitude,
                                              const double omega)
            : Sphere<dim>(sphere_radius, center_x, center_y),
             amplitude(amplitude), omega(omega) {}

    template<int dim>
    double OscillatingSphere<dim>::
    value(const Point<dim> &p, const unsigned int component) const {
        (void) component;
        double t = this->get_time();

        double x0;
        if (t <= 1) {
            x0 = amplitude * sin(omega * t) / omega - amplitude * t * cos(omega * t);
        } else {
            x0 = -amplitude * cos(omega * t);
        }
        double y0 = this->center_y;
        double x = p[0];
        double y = p[1];
        return -sqrt(pow(x - x0, 2) + pow(y - y0, 2)) + this->sphere_radius;
    }

    template<int dim>
    Tensor<1, dim> OscillatingSphere<dim>::
    get_velocity() {
        double t = this->get_time();
        Tensor<1, dim> v;
        if (t <= 1) {
            v[0] = amplitude * omega * t * sin(omega * t);
        } else {
            v[0] = amplitude * omega * sin(omega * t);
        }
        return v;
    }


    template
    class BoundaryValues<2>;

    template
    class OscillatingSphere<2>;
}
