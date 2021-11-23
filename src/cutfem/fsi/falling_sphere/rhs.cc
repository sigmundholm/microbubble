
#include "rhs.h"


namespace cut::fsi::falling_sphere {


    template<int dim>
    double cross_product(Tensor<1, dim> a, Tensor<1, dim> b) {
        return a[0] * b[1] - a[1] * b[0];
    }


    template<int dim>
    RightHandSide<dim>::RightHandSide()
            : TensorFunction<1, dim>() {}

    template<int dim>
    Tensor<1, dim> RightHandSide<dim>::
    value(const Point<dim> &p) const {
        Tensor<1, dim> gravity;
        gravity[1] = -9.81;
        return gravity;
    }


    template<int dim>
    BoundaryValues<dim>::BoundaryValues(const double half_length,
                                        const double radius,
                                        const double sphere_radius,
                                        const Tensor<1, dim> r0)
            : TensorFunction<1, dim>(), half_length(half_length),
              radius(radius), sphere_radius(sphere_radius), position(r0) {}

    template<int dim>
    Tensor<1, dim> BoundaryValues<dim>::
    value(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];

        Tensor<1, dim> zero_velocity;
        if (x == -half_length || x == half_length
            || y == -radius || y == radius) {
            // Zero Dirichlet boundary conditions on the whole boundary.
            return zero_velocity;
        }
        return velocity;
    }

    template<int dim>
    void BoundaryValues<dim>::
    set_sphere_center_velocity(Tensor<1, dim> value) {
        velocity = value;
    }

    template<int dim>
    void BoundaryValues<dim>::
    set_sphere_center_position(Tensor<1, dim> value) {
        position = value;
    }

    template<int dim>
    void BoundaryValues<dim>::
    set_sphere_angular_velocity(double value) {
        angular_velocity = value;
    }

    template<int dim>
    Tensor<1, dim> BoundaryValues<dim>::
    compute_boundary_velocity(Tensor<1, dim> point) {

        // This is the velocity component at the sphere boundary resulting from
        // the sphere rotation.
        Tensor<1, dim> angular_contrib;
        // The cross product of the angular velocity vector (in 2D this vector
        // points in the z-direction), and the position of the sphere center.
        angular_contrib[0] = angular_velocity * (point[1] - position[1]);
        angular_contrib[1] = angular_velocity * (point[0] - position[0]);

        return velocity + angular_contrib;
    }


    template<int dim>
    MovingDomain<dim>::MovingDomain(const double half_length,
                                    const double radius,
                                    const double sphere_radius,
                                    const Tensor<1, dim> r0)
            :  half_length(half_length), radius(radius),
               sphere_radius(sphere_radius), new_position(r0) {}

    template<int dim>
    double MovingDomain<dim>::
    value(const Point<dim> &p, const unsigned int component) const {
        (void) component;
        double x0 = new_position[0];
        double y0 = new_position[1];
        double x = p[0];
        double y = p[1];
        return -sqrt(pow(x - x0, 2) + pow(y - y0, 2)) + sphere_radius;
    }

    template<int dim>
    void MovingDomain<dim>::
    set_position(Tensor<1, dim> value) {
        this->new_position = value;
    }

    template
    double cross_product<2>(Tensor<1, 2>, Tensor<1, 2>);

    template
    class RightHandSide<2>;

    template
    class BoundaryValues<2>;

    template
    class MovingDomain<2>;

}
