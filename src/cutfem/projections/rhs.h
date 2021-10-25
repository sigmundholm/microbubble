#ifndef MICROBUBBLE_PROJECTIONS_RHS_H
#define MICROBUBBLE_PROJECTIONS_RHS_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include "../utils/cutfem_problem.h"

using namespace dealii;

namespace examples::cut::projections {
    using namespace utils::problems;

    template<int dim>
    class AnalyticalVelocity : public TensorFunction<1, dim> {
    public:
        Tensor<1, dim>
        value(const Point<dim> &p) const override;

        Tensor<2, dim>
        gradient(const Point<dim> &p) const override;
    };


    template<int dim>
    class AnalyticalPressure : public Function<dim> {
    public:
        double
        value(const Point<dim> &p, const unsigned int component) const override;

        Tensor<1, dim>
        gradient(const Point<dim> &p,
                 const unsigned int component) const override;
    };


    template<int dim>
    class MovingDomain : public LevelSet<dim> {
    public :
        MovingDomain(const double sphere_radius,
                     const double half_length,
                     const double radius);

        double
        value(const Point<dim> &p, const unsigned int component) const override;

        Tensor<1, dim>
        get_velocity() override;

    private:
        const double sphere_radius;
        const double half_length;
        const double radius;
    };


} // namespace examples::cut::projections


#endif // MICROBUBBLE_PROJECTIONS_RHS_H
