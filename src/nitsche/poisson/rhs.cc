#include "rhs.h"

template<int dim>
double
RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const {
    (void) p;
    return 1;
}

template<int dim>
double
BoundaryValues<dim>::value(const Point<dim> &p, const unsigned int) const {
    (void) p;
    return 0;
}


template
class RightHandSide<2>;

template
class RightHandSide<3>;

template
class BoundaryValues<2>;

template
class BoundaryValues<3>;