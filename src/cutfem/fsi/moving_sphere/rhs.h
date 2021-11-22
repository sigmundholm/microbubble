#ifndef MICROBUBBLE_RHS_H
#define MICROBUBBLE_RHS_H

#include <deal.II/base/tensor_function.h>

#include "../../utils/cutfem_problem.h"


namespace cut::fsi::moving_sphere {
    using namespace utils::problems;

    template<int dim>
    class BoundaryValues : public TensorFunction<1, dim> {
    public:
        BoundaryValues(const double half_length,
                       const double radius,
                       const double sphere_radius,
                       LevelSet<dim> &domain);

        Tensor<1, dim>
        value(const Point<dim> &p) const override;

    private:
        const double half_length;
        const double radius;
        const double sphere_radius;
        LevelSet<dim> *level_set;
    };


    template<int dim>
    class MovingDomain : public LevelSet<dim> {
    public :
        MovingDomain(double half_length,
                     double radius,
                     double sphere_radius,
                     double y_coord);

        double
        value(const Point<dim> &p, unsigned int component) const override;

        Tensor<1, dim>
        get_velocity() override;

    private:
        const double half_length;
        const double radius;
        const double sphere_radius;
        const double y_coord;
    };
}

#endif //MICROBUBBLE_RHS_H
