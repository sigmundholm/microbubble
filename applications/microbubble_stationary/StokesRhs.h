#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

using namespace dealii;

/**
 * This function is the -Laplacian
 *
 * f = -\nabla^2 u
 *
 * in \mathbb{R}^d applied to the function
 * PoissonAnalytical:
 *
 * f(r) = \omega^2 ( cos(\omega r) + (d-1)sin(\omega r)/(\omega r) ),
 *
 * where r is the distance to origo and \omega is a given frequency.
 */
template <int dim>
class PoissonRhs : public Function<dim>
{
public:
  PoissonRhs(const double frequency, const Point<dim> &center);

  double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

private:
  const double frequency;

  const Point<dim> center;
};

/**
 * This function is
 *
 * u(r) = cos(\omega R) - cos(\omega r),
 *
 * where r is the distance to origo.
 * \omega is some given frequency and R is a constant radius at which the
 * function is zero.
 */
template <int dim>
class PoissonAnalytical : public Function<dim>
{
public:
  PoissonAnalytical(const double      frequency,
                    const Point<dim> &center,
                    const double      radius_of_boundary);

  double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  Tensor<1, dim>
  gradient(const Point<dim> & point,
           const unsigned int component = 0) const override;

private:
  const double frequency;

  const Point<dim> center;

  const double radius_of_boundary;
};
