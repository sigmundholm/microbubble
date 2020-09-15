#ifndef MICROBUBBLE_ERRORRHS_H
#define MICROBUBBLE_ERRORRHS_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include "../StokesCylinder.h"


template<int dim>
class AnalyticalSolution : public TensorFunction<1, dim> {
public:
    AnalyticalSolution(const double radius, const double length,
                       const double pressure_drop, const double sphere_x_coord,
                       const double sphere_radius);

    virtual double
    point_value(const Point<dim> &p, const unsigned int component = 0) const;

    void
    vector_value(const Point<dim> &p, Tensor<1, dim> &value) const;

    void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<Tensor<1, dim>> &values) const override;

    void
    pressure_value_list(const std::vector<Point<dim>> &points,
                        std::vector<double> &values);

    double radius;
    double length;
    double pressure_drop;
    double sphere_x_coord;
    double sphere_radius;
};


template<int dim>
class ErrorStokesRhs : public StokesRhs<dim> {
public:
    ErrorStokesRhs(double radius, double length, double pressure_drop);

    double point_value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

    double radius;
    double length;
    double pressure_drop;
};


template<int dim>
class ErrorBoundaryValues : public BoundaryValues<dim> {
public:
    ErrorBoundaryValues(double radius, double length, double pressure_drop);

    double
    point_value(const Point<dim> &p,
                const unsigned int component) const override;

private:
    double pressure_drop;
};


struct Error {
    double mesh_size = 0;
    double l2_error_u = 0;
    double h1_error_u = 0;
    double l2_error_p = 0;
    double h1_error_p = 0;
};

#endif // MOCROBUBBLE_ERRORRHS_H
