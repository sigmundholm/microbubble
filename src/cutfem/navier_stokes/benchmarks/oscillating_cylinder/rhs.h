#ifndef MICROBUBBLE_NAVIER_STOKES_BM_OCILLATING_RHS_H
#define MICROBUBBLE_NAVIER_STOKES_BM_OCILLATING_RHS_H

#include <deal.II/base/tensor_function.h>

#include "../../navier_stokes.h"

namespace examples::cut::NavierStokes::benchmarks::oscillating {
    using namespace utils::problems;
    using namespace examples::cut::NavierStokes;

    template<int dim>
    class BoundaryValues : public ParabolicFlow<dim> {
    public:
        BoundaryValues(const double half_length,
                       const double radius,
                       const double sphere_radius,
                       LevelSet<dim> &domain);

        Tensor<1, dim>
        value(const Point<dim> &p) const override;

    private:
        LevelSet<dim> *level_set;
    };


    template<int dim>
    class OscillatingSphere : public Sphere<dim> {
    public :
        OscillatingSphere(double sphere_radius,
                          double center_x,
                          double center_y,
                          double amplitude,
                          double omega);

        double
        value(const Point<dim> &p, unsigned int component) const override;

        Tensor<1, dim>
        get_velocity() override;

    private:
        const double amplitude;
        const double omega;
    };
}

#endif // MICROBUBBLE_NAVIER_STOKES_BM_OCILLATING_RHS_H
