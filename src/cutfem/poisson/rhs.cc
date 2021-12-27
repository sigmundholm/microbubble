#include "rhs.h"


#define pi 3.141592653589793


template<int dim>
RightHandSide<dim>::RightHandSide(const double center_x,
                                  const double center_y)
        : center_x(center_x), center_y(center_y) {}

template<int dim>
double
RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const {
    double x = p[0] - center_x;
    double y = p[1] - center_y;
    return 2 * pi * pi * sin(pi * x) * sin(pi * y);
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
    return sin(pi * x) * sin(pi * y);
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
    return sin(pi * x) * sin(pi * y);
}

template<int dim>
Tensor<1, dim> AnalyticalSolution<dim>::
gradient(const Point<dim> &p, const unsigned int component) const {
    (void) component;
    double x = p[0] - center_x;
    double y = p[1] - center_y;
    Tensor<1, dim> value;
    value[0] = pi * sin(pi * y) * cos(pi * x);
    value[1] = pi * sin(pi * x) * cos(pi * y);

    return value;
}


template<int dim>
FlowerDomain<dim>::FlowerDomain(const double center_x, const double center_y)
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
    return sqrt(pow(x, 2) + pow(y, 2)) - r + (r / r0) * cos(5 * atan2(y, x));
}


template<int dim>
Sphere<dim>::Sphere(const double sphere_radius,
                    const double center_x,
                    const double center_y)
        : sphere_radius(sphere_radius), center_x(center_x),
            center_y(center_y) {}

template<int dim>
double Sphere<dim>::
value(const Point<dim> &p, const unsigned int component) const {
    (void) component;
    double x = p[0];
    double y = p[1];
    return sqrt(pow(x - center_x, 2) + pow(y - center_y, 2)) -
            sphere_radius;
}

template<int dim>
double Sphere<dim>::
get_radius() { return sphere_radius; }


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
class Sphere<2>;

template
class Sphere<3>;
