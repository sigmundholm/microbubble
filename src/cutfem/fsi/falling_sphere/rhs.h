#ifndef MICROBUBBLE_RHS_H
#define MICROBUBBLE_RHS_H
// TODO make the above unique.

#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

#include "../../navier_stokes/navier_stokes.h"


namespace cut::fsi::falling_sphere {
    using namespace examples::cut;
    using namespace utils::problems;

    /**
     * Compute the cross product between two vectors in two dimension, when the
     * first vector is represented as a Point object. This method is used to
     * compute the torque resulting from the fluid forces acting on a submerged
     * body.
     *
     * @tparam dim = 2 only
     * @param a
     * @param b
     * @return
     */
    template<int dim>
    double cross_product(Point<dim> a, Tensor<1, dim> b);

    template<int dim>
    class RightHandSide : public TensorFunction<1, dim> {
    public:
        RightHandSide();

        Tensor<1, dim>
        value(const Point<dim> &p) const override;
    };


    template<int dim>
    class BoundaryValues : public TensorFunction<1, dim> {
    public:
        BoundaryValues(double half_length, double radius, double sphere_radius);

        Tensor<1, dim>
        value(const Point<dim> &p) const override;

        void
        set_sphere_velocity(Tensor<1, dim> value);

        const double half_length;
        const double radius;
        const double sphere_radius;
        Tensor<1, dim> sphere_velocity;
    };


    template<int dim>
    class MovingDomain : public LevelSet<dim> {
    public :
        MovingDomain(double half_length, double radius, double sphere_radius,
                     double x0, double y0);

        double
        value(const Point<dim> &p, unsigned int component) const override;

        void
        set_position(Tensor<1, dim> value);

    private:
        const double half_length;
        const double radius;
        const double sphere_radius;

        const double x0;
        const double y0;

        Tensor<1, dim> new_position;
    };
}

#endif //MICROBUBBLE_RHS_H
