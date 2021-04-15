#ifndef MICROBUBBLE_NITSCHE_POISSON_RHS_H
#define MICROBUBBLE_NITSCHE_POISSON_RHS_H


#include <deal.II/base/function.h>
#include <deal.II/base/point.h>


using namespace dealii;

template<int dim>
class RightHandSide : public Function<dim> {
public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
};

template<int dim>
class BoundaryValues : public Function<dim> {
public:
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
};


#endif // MICROBUBBLE_NITSCHE_POISSON_RHS_H
