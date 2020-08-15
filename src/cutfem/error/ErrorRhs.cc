#include "ErrorRhs.h"


using namespace dealii;


template<int dim>
AnalyticalSolution<dim>::
AnalyticalSolution(const double radius, const double length,
                   const double pressure_drop)
        : Function<dim>(dim + 1), radius(radius), length(length),
          pressure_drop(pressure_drop) {}


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
ErrorStokesRhs(double radius, double length, double pressure_drop)
        : radius(radius), length(length), pressure_drop(pressure_drop) {}

template<int dim>
double ErrorStokesRhs<dim>::
point_value(const Point <dim> &p, const unsigned int) const {
    (void) p;
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
    if (component == 0 && p[0] == -this->length / 2) {
        if (dim == 2) {
            return pressure_drop / (4 * this->length) *
                   (this->radius * this->radius - p[1] * p[1]);
        }
        throw std::exception(); // TODO fix 3D
    }
    return 0;
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
