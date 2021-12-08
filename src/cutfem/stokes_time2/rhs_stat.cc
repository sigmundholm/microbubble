#include <cmath>

#include "rhs_stat.h"


namespace examples::cut::StokesEquation::stationary {

    template<int dim>
    RightHandSide<dim>::RightHandSide(const double nu)
            : TensorFunction<1, dim>(), nu(nu) {}

    template<int dim>
    Tensor<1, dim> RightHandSide<dim>::
    value(const Point<dim> &p) const {
        double x = p[0];
        double y = p[1];
        double t = this->get_time();

        Tensor<1, dim> val;

        val[0] = M_PI * (-4 * M_PI * nu * exp(2 * M_PI * M_PI * nu * t) * sin(M_PI * y) *
                       cos(M_PI * x) + sin(2 * M_PI * x)) *
                 exp(-4 * M_PI * M_PI * nu * t) / 2;
        val[1] = M_PI * (4 * M_PI * nu * exp(2 * M_PI * M_PI * nu * t) * sin(M_PI * x) *
                       cos(M_PI * y) + sin(2 * M_PI * y)) *
                 exp(-4 * M_PI * M_PI * nu * t) / 2;
        return val;
    }


    template
    class RightHandSide<2>;

}