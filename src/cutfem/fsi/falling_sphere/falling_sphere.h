#ifndef MICROBUBBLE_EX_FALLING_SPHERE_H
#define MICROBUBBLE_EX_FALLING_SPHERE_H

// #include "../../navier_stokes/navier_stokes.h"

#include <boost/optional.hpp>
#include <queue>

#include "rhs.h"


namespace cut::fsi::falling_sphere {
    // using namespace examples::cut;

    template<int dim>
    class FallingSphere : public NavierStokes::NavierStokesEqn<dim> {
    public:
        FallingSphere(double nu, double tau, double radius,
                      double half_length, unsigned int n_refines,
                      int element_order, bool write_output,
                      TensorFunction<1, dim> &rhs,
                      TensorFunction<1, dim> &bdd_values,
                      TensorFunction<1, dim> &analytic_vel,
                      Function<dim> &analytic_pressure,
                      LevelSet<dim> &levelset_func,
                      double fluid_density,
                      double sphere_density,
                      double sphere_radius,
                      bool semi_implicit,
                      int do_nothing_id = 10);

    protected:

        void
        pre_time_loop(unsigned int bdf_type, unsigned int steps) override;

        void
        post_processing(unsigned int time_step) override;

        void
        conservation_linear_momentum();

        void
        conservation_angular_momentum();

        void
        update_boundary_values();

        /**
         * This method computes the torque resulting from the viscous and pressure
         * forces on the submerged body, in two dimensions.
         *
         * NB: this only works for dim = 2.
         *
         * @tparam dim = 2 only
         * @return
         */
        double
        compute_surface_torque();

        void
        integrate_surface_torque(const FEValuesBase<dim> &fe_v,
                                 Vector<double> solution,
                                 double &viscous_torque,
                                 double &pressure_torque);

        const double fluid_density;
        const double sphere_density;
        const double sphere_radius;

        MovingDomain<dim> *domain;
        BoundaryValues<dim> *boundary;

        boost::optional<std::deque<Tensor<1, dim>>> positions;
        boost::optional<std::deque<Tensor<1, dim>>> velocities;
        boost::optional<std::deque<Tensor<1, dim>>> accelerations;
        boost::optional<std::deque<double>> angles;
        boost::optional<std::deque<double>> angular_velocities;
        boost::optional<std::deque<double>> angular_accelerations;

        std::ofstream file;
    };
}


#endif //MICROBUBBLE_EX_FALLING_SPHERE_H
