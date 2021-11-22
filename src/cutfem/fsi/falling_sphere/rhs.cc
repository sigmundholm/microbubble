
#include "rhs.h"


namespace cut::fsi::falling_sphere {


    template<int dim>
    double cross_product(Point<dim> a, Tensor<1, dim> b) {
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
                                        const double sphere_radius)
            : TensorFunction<1, dim>(), half_length(half_length),
              radius(radius), sphere_radius(sphere_radius) {}

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
        return sphere_velocity;
    }

    template<int dim>
    void BoundaryValues<dim>::
    set_sphere_velocity(Tensor<1, dim> value) {
        sphere_velocity = value;
    }


    template<int dim>
    MovingDomain<dim>::MovingDomain(const double half_length,
                                    const double radius,
                                    const double sphere_radius,
                                    const double x0,
                                    const double y0)
            :  half_length(half_length), radius(radius),
               sphere_radius(sphere_radius), x0(x0), y0(y0) {}

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
    double cross_product<2>(Point<2>, Tensor<1, 2>);

    template
    class RightHandSide<2>;

    template
    class BoundaryValues<2>;

    template
    class MovingDomain<2>;

}
