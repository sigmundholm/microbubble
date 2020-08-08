#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometric_utilities.h>

#include <boost/math/special_functions/sinc.hpp>

#include <cmath>
#include <iostream>

#include "StokesRhs.h"

template <int dim>
StokesRhs<dim>::StokesRhs(const double frequency, const Point<dim> &center)
  : frequency(frequency)
  , center(center)
{}



template <int dim>
double
StokesRhs<dim>::value(const Point<dim> &p, const unsigned int component) const
{
  Assert(component == 0, ExcInternalError());

  using boost::math::sinc_pi;

  const double radius = p.distance(center);
  const double value =
    -frequency * frequency *
    (std::cos(frequency * radius) + (dim - 1) * sinc_pi(frequency * radius));

  return value;
}



template <int dim>
StokesAnalytical<dim>::StokesAnalytical(const double      frequency,
                                          const Point<dim> &center,
                                          const double      radius_of_boundary)
  : frequency(frequency)
  , center(center)
  , radius_of_boundary(radius_of_boundary)
{}



template <int dim>
double
StokesAnalytical<dim>::value(const Point<dim> & p,
                              const unsigned int component) const
{
  Assert(component == 0, ExcInternalError());

  using boost::math::sinc_pi;

  const double radius = p.distance(center);
  const double value =
    (std::cos(frequency * radius_of_boundary) - std::cos(frequency * radius));

  return value;
}



template <int dim>
Tensor<1, dim>
StokesAnalytical<dim>::gradient(const Point<dim> & point,
                                 const unsigned int component) const
{
  Assert(component == 0, ExcInternalError());

  const double radius = point.distance(center);
  const double dudr   = frequency * std::sin(frequency * radius);

  Point<dim> relative_center = point;
  relative_center -= center;

  Tensor<1, dim> gradient = dudr * relative_center / relative_center.norm();
  return gradient;
}

template class StokesRhs<2>;
template class StokesRhs<3>;

template class StokesAnalytical<2>;
template class StokesAnalytical<3>;
