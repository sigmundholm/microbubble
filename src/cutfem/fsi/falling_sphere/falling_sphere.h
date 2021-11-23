#ifndef MICROBUBBLE_FSI_FALLING_SPHERE_H
#define MICROBUBBLE_FSI_FALLING_SPHERE_H

#include <boost/optional.hpp>
#include <queue>

#include "rhs.h"


namespace cut::fsi::falling_sphere {

    /**
     * This class creates a simulation of a sphere submerged in a fluid, and
     * and the forces acting on the sphere, causes it either to sink, or float.
     *
     * The fluid is modelled using Navier-Stokes equations, while the sphere
     * moves as a rigid body. The forces acting on the sphere is gravity and
     * friction forces acting on the sphere surfaces resulting from the viscous
     * forces and pressure forces from the fluid on the sphere.
     *
     * To run the simulation, se the method run_moving_domain(bdf_type, steps).
     * Then in each time step, when the method post_processing() of this class
     * is called, the forces on the sphere is computed, and its position for the
     * next step is calculated by solving the ODEs resulting from conservation
     * of linear and angular momentum.
     *
     * @tparam dim
     */
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
                      Tensor<1, dim> r0,
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

        void
        write_data(unsigned int time_step);

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

        Tensor<1, dim> r0;

        boost::optional<std::deque<Tensor<1, dim>>> positions;
        boost::optional<std::deque<Tensor<1, dim>>> velocities;
        boost::optional<std::deque<Tensor<1, dim>>> accelerations;
        boost::optional<std::deque<double>> angles;
        boost::optional<std::deque<double>> angular_velocities;
        boost::optional<std::deque<double>> angular_accelerations;

        std::ofstream file;
    };
}


#endif // MICROBUBBLE_FSI_FALLING_SPHERE_H
