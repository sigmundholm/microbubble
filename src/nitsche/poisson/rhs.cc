#include <math.h>
#include <iostream>

#include "rhs.h"

template<int dim>
double
RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const {
    double x = p[0];
    double y = p[1];
    return - 2 * M_PI * M_PI * cos(M_PI * x) * sin(M_PI * y);
}

template<int dim>
double
BoundaryValues<dim>::value(const Point<dim> &p, const unsigned int) const {
    double x = p[0];
    double y = p[1];
    return cos(M_PI * x) * sin(M_PI * y);
}


template<int dim>
double AnalyticalSolution<dim>::
value(const Point<dim> &p, const unsigned int component) const {
    (void) component;
    double x = p[0];
    double y = p[1];
    return cos(M_PI * x) * sin(M_PI * y);
}

template<int dim>
Tensor<1, dim> AnalyticalSolution<dim>::
gradient(const Point<dim> &p, const unsigned int component) const {
    (void) component;
    double x = p[0];
    double y = p[1];
    Tensor<1, dim> value;
    value[0] = - M_PI * sin(M_PI * x) * sin(M_PI * y);
    value[1] = M_PI * cos(M_PI * x) * cos(M_PI * y);
    return value;
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
