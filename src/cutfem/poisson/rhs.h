#ifndef MICROBUBBLE_CUTFEM_POISSON_RHS_H
#define MICROBUBBLE_CUTFEM_POISSON_RHS_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>


using namespace dealii;

template<int dim>
class RightHandSide : public Function<dim> {
public:
    RightHandSide(const double center_x = 0,
                  const double center_y = 0);

    double
    value(const Point<dim> &p,
          const unsigned int component = 0) const override;

private:
    const double center_x;
    const double center_y;
};

template<int dim>
class BoundaryValues : public Function<dim> {
public:
    BoundaryValues(const double center_x = 0,
                   const double center_y = 0);

    double
    value(const Point<dim> &p, const unsigned int component = 0) const override;

private:
    const double center_x;
    const double center_y;
};


template<int dim>
class AnalyticalSolution : public Function<dim> {
public:
    AnalyticalSolution(const double center_x = 0,
                       const double center_y = 0);

    double
    value(const Point<dim> &p, const unsigned int component) const override;

    Tensor<1, dim>
    gradient(const Point<dim> &p, const unsigned int component) const override;

private:
    const double center_x;
    const double center_y;
};


template<int dim>
class FlowerDomain : public Function<dim> {
public :
    FlowerDomain(const double center_x = 0,
                 const double center_y = 0);

    double
    value(const Point<dim> &p, const unsigned int component) const override;

private:
    const double center_x;
    const double center_y;
};


#endif //MICROBUBBLE_CUTFEM_POISSON_RHS_H
