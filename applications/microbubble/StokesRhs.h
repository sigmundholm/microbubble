#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

using namespace dealii;


template <int dim>
class StokesRhs : public TensorFunction<1, dim>
{
public:
  virtual double
  point_value(const Point<dim> &p, const unsigned int component = 0) const;

  void
  vector_value(const Point<dim> &p, Tensor<1, dim> &value) const;

  void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<Tensor<1, dim>> &  values) const override;
};


template <int dim>
class BoundaryValues : public TensorFunction<1, dim>
{
public:
  BoundaryValues(double radius, double length);

  double
  point_value(const Point<dim> &p, const unsigned int component) const;

  void
  vector_value(const Point<dim> &p, Tensor<1, dim> &value) const;

  void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<Tensor<1, dim>> &  values) const;

private:
  double radius;
  double length;
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
class StokesAnalytical : public Function<dim>
{
public:
  StokesAnalytical(const double      frequency,
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
