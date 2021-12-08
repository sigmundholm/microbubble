#ifndef MICROBUBBLE_STOKES_TIME2_RHS_STAT_H
#define MICROBUBBLE_STOKES_TIME2_RHS_STAT_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

using namespace dealii;

namespace examples::cut::StokesEquation::stationary {

    template<int dim>
    class RightHandSide : public TensorFunction<1, dim> {
    public:
        RightHandSide(double nu);

        Tensor<1, dim>
        value(const Point<dim> &p) const override;

        const double nu;
    };
}

#endif //MICROBUBBLE_STOKES_TIME2_RHS_STAT_H
