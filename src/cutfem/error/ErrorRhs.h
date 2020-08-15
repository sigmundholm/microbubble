#ifndef MICROBUBBLE_ERRORRHS_H
#define MICROBUBBLE_ERRORRHS_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include "../StokesCylinder.h"


template<int dim>
class AnalyticalSolution : public Function<dim> {
public:
    AnalyticalSolution(const double radius, const double length,
                       const double pressure_drop);

    void vector_value(const Point <dim> &p,
                      Vector<double> &value) const override;

    double radius;
    double length;
    double pressure_drop;
};


template<int dim>
class ErrorStokesRhs : public StokesRhs<dim> {
public:
    ErrorStokesRhs(double length, double radius, double pressure_drop);

    double point_value(const Point <dim> &p,
                       const unsigned int component = 0) const override;

    double length;
    double radius;
    double pressure_drop;
};


template<int dim>
class ErrorBoundaryValues : public BoundaryValues<dim> {
public:
    ErrorBoundaryValues(double radius, double length);

    double
    point_value(const Point <dim> &p, const unsigned int component) const;
};


#endif // MOCROBUBBLE_ERRORRHS_H
