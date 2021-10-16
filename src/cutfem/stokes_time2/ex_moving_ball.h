#ifndef MICROBUBBLE_EX_MOVING_BALL_H
#define MICROBUBBLE_EX_MOVING_BALL_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>


using namespace dealii;

namespace examples::cut::StokesEquation::ex1 {

    template<int dim>
    class BoundaryValues : public TensorFunction<1, dim> {
    public:
        BoundaryValues(const double sphere_radius, const double half_length,
                       const double radius);

        Tensor<1, dim>
        value(const Point<dim> &p) const override;

        const double sphere_radius;
        const double half_length;
        const double radius;
    };


    template<int dim>
    class MovingDomain : public Function<dim> {
    public :
        MovingDomain(const double sphere_radius,
                     const double half_length,
                     const double radius);

        double
        value(const Point<dim> &p, const unsigned int component) const override;

    private:
        const double sphere_radius;
        const double half_length;
        const double radius;
    };


}


#endif //MICROBUBBLE_EX_MOVING_BALL_H
