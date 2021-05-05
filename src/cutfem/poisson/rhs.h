#ifndef MICROBUBBLE_CUTFEM_POISSON_RHS_H
#define MICROBUBBLE_CUTFEM_POISSON_RHS_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>


struct Error {
    double mesh_size = 0;
    double l2_error = 0;
    double h1_error = 0;
    double h1_semi = 0;
};


using namespace dealii;

template<int dim>
class RightHandSide : public Function<dim> {
public:
    double
    value(const Point<dim> &p,
          const unsigned int component = 0) const override;
};

template<int dim>
class BoundaryValues : public Function<dim> {
public:
    double
    value(const Point<dim> &p,
          const unsigned int component = 0) const override;
};


template<int dim>
class AnalyticalSolution : public Function<dim> {
public:
    double
    value(const Point<dim> &p, const unsigned int component) const override;

    Tensor<1, dim>
    gradient(const Point<dim> &p, const unsigned int component) const override;
};


#endif //MICROBUBBLE_CUTFEM_POISSON_RHS_H
