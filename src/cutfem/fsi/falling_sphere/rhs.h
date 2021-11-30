#ifndef MICROBUBBLE_FSI_FALLING_SPHERE_RHS_H
#define MICROBUBBLE_FSI_FALLING_SPHERE_RHS_H
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
    double cross_product(Tensor<1, dim> a, Tensor<1, dim> b);

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
        BoundaryValues(double half_length, double radius, double sphere_radius,
                       Tensor<1, dim> r0);

        Tensor<1, dim>
        value(const Point<dim> &p) const override;

        void
        set_sphere_center_velocity(Tensor<1, dim> value);

        void
        set_sphere_center_position(Tensor<1, dim> value);

        void
        set_sphere_angular_velocity(double value);

        /**
         * This compute the value for the no-slip boundary condition around the
         * sphere. The given point is a point at the boundary of the sphere.
         * The velocity at that point is the sum of the contribution from the
         * linear movement of the spheres center, and the velocity at that
         * point resulting from the rotation of the sphere. That is,
         *
         *   u(x, t) = V(x, t) +  ω x (point - r_s)
         *
         * This computations is done for 2D only. In 2D, ω is the angular
         * momentum as a vector always pointing along the z-axis. This causes
         * the cross product above to lie in the xy-plane.
         */
        Tensor<1, dim>
        compute_boundary_velocity(Tensor<1, dim> point);

        const double half_length;
        const double radius;
        const double sphere_radius;

        // The computed position of the sphere center in the next time step.
        Tensor<1, dim> position;
        // The velocity of the center of the sphere in the next time step.
        Tensor<1, dim> velocity;
        // The angular velocity of the sphere in the next time step.
        double angular_velocity;
    };


    template<int dim>
    class MovingDomain : public LevelSet<dim> {
    public :
        MovingDomain(double half_length, double radius, double sphere_radius,
                     Tensor<1, dim> r0);

        double
        value(const Point<dim> &p, unsigned int component) const override;

        void
        set_position(Tensor<1, dim> value);

        void
        set_velocity(Tensor<1, dim> value);

        Tensor<1, dim>
        get_velocity() override;

    private:
        const double half_length;
        const double radius;
        const double sphere_radius;

        Tensor<1, dim> position;
        Tensor<1, dim> velocity;
    };
}

#endif // MICROBUBBLE_FSI_FALLING_SPHERE_RHS_H
