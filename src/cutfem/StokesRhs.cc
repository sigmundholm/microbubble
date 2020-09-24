#include "StokesRhs.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometric_utilities.h>

#include <boost/math/special_functions/sinc.hpp>

#include <iostream>

template<int dim>
double
StokesRhs<dim>::point_value(const Point<dim> &p, const unsigned int) const {
    (void) p;
    return 0;
}

template<int dim>
void
StokesRhs<dim>::vector_value(const Point<dim> &p, Tensor<1, dim> &value) const {
    for (unsigned int c = 0; c < dim; ++c)
        value[c] = point_value(p, c);
}

template<int dim>
void
StokesRhs<dim>::value_list(const std::vector<Point<dim>> &points,
                           std::vector<Tensor<1, dim>> &values) const {
    AssertDimension(points.size(), values.size());
    for (unsigned int i = 0; i < values.size(); ++i) {
        vector_value(points[i], values[i]);
    }
}


template<int dim>
BoundaryValues<dim>::BoundaryValues(double radius, double length)
        : radius(radius), length(length) {}

template<int dim>
double
BoundaryValues<dim>::point_value(const Point<dim> &p,
                                 const unsigned int component) const {
    (void) p;
    double pressure_drop = 10;  // Only for cylinder channel, Hagen–Poiseuille
    if (component == 0 && p[0] == -length / 2) {
        if (dim == 2) {
            return pressure_drop / (4 * length) *
                   (radius * radius - p[1] * p[1]);
        }
        throw std::exception(); // TODO fix 3D
    }
    return 0;
}

template<int dim>
Tensor<1, dim>
BoundaryValues<dim>::value(const Point<dim> &p) const {
    // Overridden method, used in VectorFunctionFromTensorFunction, when
    // interpolating the initial values from this class.
    Tensor<1, dim> val;
    for (unsigned int c = 0; c < dim; ++c) {
        val[c] = point_value(p, c);
    }
    return val;
}

template<int dim>
void
BoundaryValues<dim>::vector_value(const Point<dim> &p,
                                  Tensor<1, dim> &value) const {
    for (unsigned int c = 0; c < dim; ++c)
        value[c] = point_value(p, c);
}

template<int dim>
void
BoundaryValues<dim>::value_list(const std::vector<Point<dim>> &points,
                                std::vector<Tensor<1, dim>> &values) const {
    AssertDimension(points.size(), values.size());
    for (unsigned int i = 0; i < values.size(); ++i) {
        vector_value(points[i], values[i]);
    }
}


template
class StokesRhs<2>;

template
class StokesRhs<3>;

template
class BoundaryValues<2>;

template
class BoundaryValues<3>;
