#include "ErrorRhs.h"


using namespace dealii;


template<int dim>
AnalyticalSolution<dim>::
AnalyticalSolution(const double radius, const double length,
                   const double pressure_drop)
        : radius(radius), length(length), pressure_drop(pressure_drop) {}


template<int dim>
void AnalyticalSolution<dim>::
vector_value(const Point <dim> &p, Vector<double> &value) const {
    Assert(value.size() == dim + 1,
           ExcDimensionMismatch(value.size(), dim + 1));
    // values = u_1, u_2, p
    value(0) = pressure_drop / (4 * length)
                * (radius * radius - p[1] * p[1]);
    value(1) = 0;
    value(2) = 0;
}


template<int dim>
ErrorStokesRhs<dim>::
ErrorStokesRhs(double length, double radius, double pressure_drop)
        : length(length), radius(radius), pressure_drop(pressure_drop) {}

template<int dim>
double ErrorStokesRhs<dim>::
point_value(const Point <dim> &p, const unsigned int component) const {
    // TODO sett
    return 0;
}


template<int dim>
ErrorBoundaryValues<dim>::
ErrorBoundaryValues(double radius, double length)
        : BoundaryValues<dim>(radius, length) {}

template<int dim>
double ErrorBoundaryValues<dim>::
point_value(const Point <dim> &p, const unsigned int component) const {
    // TODO sett
    return 0;
}


