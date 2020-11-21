#include "ErrorRhs.h"

#include <math.h>

#define PI 3.141592653589793
#define T 0

using namespace dealii;


template<int dim>
AnalyticalSolution<dim>::
AnalyticalSolution(const double radius, const double length,
                   const double pressure_drop, const double sphere_x_coord,
                   const double sphere_radius)
        : radius(radius), length(length), pressure_drop(pressure_drop),
          sphere_x_coord(sphere_x_coord), sphere_radius(sphere_radius) {}


template<int dim>
double AnalyticalSolution<dim>::
point_value(const Point<dim> &p,
            const unsigned int component) const {
    /**
     * This is an analytical solution to the Stokes equation in 2D, see Ethier
     * and Steinman (1994) equation (1).
     *
     * This also work for channel with sphere, when one only integrate the error
     * over the inside-cells.
     */
    // values = u_1, u_2, p
    if (component == 0) {
        // Velocity in x-direction
        return -cos(PI * p[0]) * sin(PI * p[1]) * exp(-2 * PI * PI * T);
    } else if (component == 1) {
        // Velocity in y-direction
        return sin(PI * p[0]) * cos(PI * p[1]) * exp(-2 * PI * PI * T);
    } else if (component == 2) {
        // Pressure component
        return -(cos(2 * PI * p[0]) + cos(2 * PI * p[1])) / 4 *
               exp(-4 * PI * PI * T);
    } else {
        throw std::exception();
    }
}

template<int dim>
void AnalyticalSolution<dim>::
vector_value(const Point<dim> &p,
             Tensor<1, dim> &value) const {
    for (unsigned int c = 0; c < dim; ++c)
        value[c] = point_value(p, c);
}

template<int dim>
void AnalyticalSolution<dim>::
value_list(const std::vector<Point<dim>> &points,
           std::vector<Tensor<1, dim>> &values) const {
    AssertDimension(points.size(), values.size());
    for (unsigned int i = 0; i < values.size(); ++i) {
        vector_value(points[i], values[i]);
    }
}

template<int dim>
void AnalyticalSolution<dim>::
gradient(const Point<dim> &p,
         Tensor<2, dim> &value) const {
    // u_x
    value[0][0] = PI * exp(-2 * PI * PI * T) * sin(PI * p[0]) * sin(PI * p[1]);
    // u_y
    value[0][1] = -PI * exp(-2 * PI * PI * T) * cos(PI * p[0]) * cos(PI * p[1]);
    // v_x
    value[1][0] = PI * exp(-2 * PI * PI * T) * cos(PI * p[0]) * cos(PI * p[1]);
    // v_y
    value[1][1] = -PI * exp(-2 * PI * PI * T) * sin(PI * p[0]) * sin(PI * p[1]);
}

template<int dim>
void AnalyticalSolution<dim>::
gradient_list(const std::vector<Point<dim>> &points,
              std::vector<Tensor<2, dim>> &values) const {
    for (unsigned int i = 0; i < values.size(); ++i) {
        gradient(points[i], values[i]);
    }
}

template<int dim>
void AnalyticalSolution<dim>::
pressure_gradient(const Point<dim> &p,
                  Tensor<1, dim> &value) const {
    value[0] = PI * exp(-4 * PI * PI * T) * sin(2 * PI * p[0]) / 2;
    value[1] = PI * exp(-4 * PI * PI * T) * sin(2 * PI * p[1]) / 2;
}

template<int dim>
void AnalyticalSolution<dim>::
pressure_gradient_list(const std::vector<Point<dim>> &points,
                       std::vector<Tensor<1, dim>> &values) const {
    for (unsigned int i = 0; i < values.size(); ++i) {
        pressure_gradient(points[i], values[i]);
    }
}

template<int dim>
void AnalyticalSolution<dim>::
pressure_value_list(const std::vector<Point<dim>> &points,
                    std::vector<double> &values) {
    AssertDimension(points.size(), values.size());
    for (unsigned int i = 0; i < values.size(); ++i) {
        values[i] = point_value(points[i], dim);
    }
}


template<int dim>
ErrorStokesRhs<dim>::
ErrorStokesRhs(double radius, double length, double pressure_drop)
        : radius(radius), length(length), pressure_drop(pressure_drop) {}

template<int dim>
double ErrorStokesRhs<dim>::
point_value(const Point<dim> &p, const unsigned int component) const {
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
}

template<int dim>
ErrorBoundaryValues<dim>::
ErrorBoundaryValues(double radius, double length, double pressure_drop)
        : BoundaryValues<dim>(radius, length), pressure_drop(pressure_drop) {}

template<int dim>
double ErrorBoundaryValues<dim>::
point_value(const Point<dim> &p, const unsigned int component) const {
    (void) p;
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


template<int dim>
double AnalyticalPressure<dim>::
value(const Point<dim> &p, const unsigned int component) const {
    (void) component;
    double x = p[0];
    double y = p[1];
    return -cos(2 * PI * x) / 4 - cos(2 * PI * y) / 4;
}


template<int dim>
Tensor<1, dim> AnalyticalPressure<dim>::
gradient(const Point<dim> &p, const unsigned int component) const {
    (void) component;
    double x = p[0];
    double y = p[1];
    Tensor<1, dim> value;
    value[0] = PI * sin(2 * PI * x) / 2;
    value[1] = PI * sin(2 * PI * y) / 2;
    return value;
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

template
class AnalyticalPressure<2>;

template
class AnalyticalPressure<3>;
