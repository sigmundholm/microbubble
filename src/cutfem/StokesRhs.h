#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

using namespace dealii;


template<int dim>
class StokesRhs : public TensorFunction<1, dim> {
public:
    virtual double
    point_value(const Point<dim> &p, const unsigned int component = 0) const;

    void
    vector_value(const Point<dim> &p, Tensor<1, dim> &value) const;

    void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<Tensor<1, dim>> &values) const override;
};


template<int dim>
class BoundaryValues : public TensorFunction<1, dim> {
public:
    BoundaryValues(double radius, double length);

    double
    point_value(const Point<dim> &p, const unsigned int component) const;

    void
    vector_value(const Point<dim> &p, Tensor<1, dim> &value) const;

    void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<Tensor<1, dim>> &values) const;

private:
    double radius;
    double length;
};
