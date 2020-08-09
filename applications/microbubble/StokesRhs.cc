#include "StokesRhs.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometric_utilities.h>

#include <boost/math/special_functions/sinc.hpp>

#include <cmath>
#include <iostream>

template <int dim>
double
StokesRhs<dim>::point_value(const Point<dim> &p, const unsigned int) const
{
  (void)p;
  return 0;
}

template <int dim>
void
StokesRhs<dim>::vector_value(const Point<dim> &p, Tensor<1, dim> &value) const
{
  for (unsigned int c = 0; c < dim; ++c)
    value[c] = point_value(p, c);
}

template <int dim>
void
StokesRhs<dim>::value_list(const std::vector<Point<dim>> &points,
                           std::vector<Tensor<1, dim>> &  values) const
{
  AssertDimension(points.size(), values.size());
  for (unsigned int i = 0; i < values.size(); ++i)
    {
      vector_value(points[i], values[i]);
    }
}



template <int dim>
BoundaryValues<dim>::BoundaryValues(double radius, double length)
  : radius(radius)
  , length(length)
{}

template <int dim>
double
BoundaryValues<dim>::point_value(const Point<dim> & p,
                                 const unsigned int component) const
{
  (void)p;
  double pressure_drop = 10;  // Only for cylinder channel, Hagen–Poiseuille
  if (component == 0 && p[0] == -length/2)
    {
      if (dim == 2)
        {
          return pressure_drop / (4 * length) * (radius * radius - p[1] * p[1]);
        }
      throw std::exception(); // TODO fix 3D
    }
  return 0;
}

template <int dim>
void
BoundaryValues<dim>::vector_value(const Point<dim> &p,
                                  Tensor<1, dim> &  value) const
{
  for (unsigned int c = 0; c < dim; ++c)
    value[c] = point_value(p, c);
}

template <int dim>
void
BoundaryValues<dim>::value_list(const std::vector<Point<dim>> &points,
                                std::vector<Tensor<1, dim>> &  values) const
{
  AssertDimension(points.size(), values.size());
  for (unsigned int i = 0; i < values.size(); ++i)
    {
      vector_value(points[i], values[i]);
    }
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

template class BoundaryValues<2>;
template class BoundaryValues<3>;

template class StokesAnalytical<2>;
template class StokesAnalytical<3>;
