#include "ErrorRhs.h"

#include <math.h>

#define PI 3.14159265
#define T 0

using namespace dealii;


template<int dim>
AnalyticalSolution<dim>::
AnalyticalSolution(const double radius, const double length,
                   const double pressure_drop, const double sphere_x_coord,
                   const double sphere_radius)
        : Function<dim>(dim + 1), radius(radius), length(length),
          pressure_drop(pressure_drop), sphere_x_coord(sphere_x_coord),
          sphere_radius(sphere_radius) {}


template<int dim>
void AnalyticalSolution<dim>::
vector_value(const Point <dim> &p, Vector<double> &value) const {
    Assert(value.size() == dim + 1,
           ExcDimensionMismatch(value.size(), dim + 1));
    // values = u_1, u_2, p
    if (pow(p[0] - sphere_x_coord, 2) + pow(p[1], 2) < pow(sphere_radius, 2)) {
        // The solution is valued 0 inside the sphere.
        value(0) = 0;
        value(1) = 0;
        value(2) = 0;
    } else {
        value(0) = -cos(PI * p[0]) * sin(PI * p[1]) * exp(-2 * PI * PI * T);
        value(1) = sin(PI * p[0]) * cos(PI * p[1]) * exp(-2 * PI * PI * T);
        value(2) = -(cos(2 * PI * p[0]) + cos(2 * PI * p[1])) / 4 *
                   exp(-4 * PI * PI * T);
    }
}


template<int dim>
ErrorStokesRhs<dim>::
ErrorStokesRhs(double radius, double length, double pressure_drop)
        : radius(radius), length(length), pressure_drop(pressure_drop) {}

template<int dim>
double ErrorStokesRhs<dim>::
point_value(const Point <dim> &p, const unsigned int component) const {
    double x = p[0];
    double y = p[1];

    if (component == 0) {
        // -(u_t + u * u_x + v * u_y)
        return -(2 * PI * PI * exp(-2 * PI * PI * T) * sin(PI * y) *
                 cos(PI * x) -
                 PI * exp(-4 * PI * PI * T) * sin(PI * x) * sin(PI * y) *
                 sin(PI * y) * cos(PI * x) -
                 PI * exp(-4 * PI * PI * T) * sin(PI * x) * cos(PI * x) *
                 cos(PI * y) * cos(PI * y));
    } else if (component == 1) {
        // -(v_t + u*v_x + v * v_y)
        return -(-2 * PI * PI * exp(-2 * PI * PI * T) * sin(PI * x) *
                 cos(PI * y) -
                 PI * exp(-4 * PI * PI * T) * sin(PI * x) * sin(PI * x) *
                 sin(PI * y) * cos(PI * y) -
                 PI * exp(-4 * PI * PI * T) * sin(PI * y) * cos(PI * x) *
                 cos(PI * x) * cos(PI * y));
    } else {
        // These expressions are only for 2D.
        throw std::exception();
    }
    return 0;
}

template<int dim>
ErrorBoundaryValues<dim>::
ErrorBoundaryValues(double radius, double length, double pressure_drop)
        : BoundaryValues<dim>(radius, length), pressure_drop(pressure_drop) {}

template<int dim>
double ErrorBoundaryValues<dim>::
point_value(const Point <dim> &p, const unsigned int component) const {
    (void) p;
    double pressure_drop = 10;  // Only for cylinder channel, Hagen–Poiseuille
    if (dim == 2) {
        if (component == 0) {
            return -cos(PI * p[0]) * sin(PI * p[1]) * exp(-2 * PI * PI * T);
        } else if (component == 1) {
            return sin(PI * p[0]) * cos(PI * p[1]) * exp(-2 * PI * PI * T);
        } else {
            throw std::exception();
        }
    } else {
        throw std::exception();
    }
}


template
class AnalyticalSolution<2>;

template
class AnalyticalSolution<3>;

template
class ErrorStokesRhs<2>;

template
class ErrorStokesRhs<3>;

template
class ErrorBoundaryValues<2>;

template
class ErrorBoundaryValues<3>;
