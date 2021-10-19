#ifndef MICROBUBBLE_EX_FALLING_SPHERE_H
#define MICROBUBBLE_EX_FALLING_SPHERE_H

#include <deal.II/base/tensor.h>

#include "stokes.h"


namespace examples::cut::StokesEquation::ex2 {

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
        BoundaryValues(const double sphere_radius, const double half_length,
                       const double radius);

        Tensor<1, dim>
        value(const Point<dim> &p) const override;

        void
        set_ball_velocity(Tensor<1, dim> value);

        const double sphere_radius;
        const double half_length;
        const double radius;
        Tensor<1, dim> ball_velocity;
    };


    template<int dim>
    class MovingDomain : public LevelSet<dim> {
    public :
        MovingDomain(const double sphere_radius,
                     const double half_length,
                     const double radius,
                     const double tau);


        double
        value(const Point<dim> &p, const unsigned int component) const override;

        void
        set_acceleration(Tensor<1, dim> value);

        double
        get_volume();

        void
        update_position();

        Tensor<1, dim>
        get_velocity() override;

    private:
        const double sphere_radius;
        const double half_length;
        const double radius;
        const double tau;

        Tensor<1, dim> last_position;
        Tensor<1, dim> new_position;

        Tensor<1, dim> last_velocity;
        Tensor<1, dim> new_velocity;

        // The acceleration at the current time step.
        Tensor<1, dim> acceleration;
    };


    template<int dim>
    class FallingSphereStokes : public StokesEqn<dim> {
    public:
        FallingSphereStokes(double nu, double tau, double radius,
                            double half_length, unsigned int n_refines,
                            int element_order, bool write_output,
                            TensorFunction<1, dim> &rhs,
                            TensorFunction<1, dim> &bdd_values,
                            TensorFunction<1, dim> &analytic_vel,
                            Function<dim> &analytic_pressure,
                            LevelSet<dim> &levelset_func,
                            const double density_ratio,
                            int do_nothing_id = 10);

    protected:
        void
        post_processing() override;

        Tensor<1, dim>
        compute_surface_forces();

        void
        integrate_surface_forces(const FEValuesBase<dim> &fe_v,
                                 Vector<double> solution,
                                 Tensor<1, dim> &viscous_forces,
                                 Tensor<1, dim> &pressure_forces);


        // The ratio between the density of the fluid and the moving sphere.
        const double density_ratio;

        std::ofstream file;
    };


}


#endif //MICROBUBBLE_EX_FALLING_SPHERE_H
