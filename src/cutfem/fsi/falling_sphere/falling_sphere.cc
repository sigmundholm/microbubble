#include <deal.II/non_matching/fe_values.h>

#include <cmath>

#include "falling_sphere.h"
#include "rhs.h"

using namespace dealii;
using namespace cutfem;

namespace cut::fsi::falling_sphere {

    template<int dim>
    FallingSphere<dim>::
    FallingSphere(const double nu, const double tau, const double radius,
                  const double half_length, const unsigned int n_refines,
                  const int element_order, const bool write_output,
                  TensorFunction<1, dim> &rhs,
                  TensorFunction<1, dim> &bdd_values,
                  TensorFunction<1, dim> &analytic_vel,
                  Function<dim> &analytic_pressure,
                  LevelSet<dim> &levelset_func,
                  const double fluid_density,
                  const double sphere_density,
                  const double sphere_radius,
                  const Tensor<1, dim> r0,
                  const bool semi_implicit,
                  const int do_nothing_id)
            : NavierStokes::NavierStokesEqn<dim>(
            nu, tau, radius, half_length, n_refines, element_order,
            write_output, rhs, bdd_values, analytic_vel, analytic_pressure,
            levelset_func, semi_implicit, do_nothing_id, true, false, false),
              fluid_density(fluid_density), sphere_density(sphere_density),
              sphere_radius(sphere_radius), r0(r0) {

        domain = dynamic_cast<MovingDomain<dim> *>(this->levelset_function);
        boundary = dynamic_cast<BoundaryValues<dim> *>(this->boundary_values);

        file = std::ofstream("fsi_falling_sphere.csv");
        file << "k; t; r_x; r_y; v_x; v_y; a_x; a_y; "
                " \\theta; \\omega; \\alpha" << std::endl;
                // "F_gx;F_gy;F_px;F_py;F_vx;F_vy" << std::endl;
    }


    template<int dim>
    void FallingSphere<dim>::
    pre_time_loop(unsigned int bdf_type, unsigned int steps) {
        // TODO overfør gammel data fra listene hvis de finnes.

        std::deque<Tensor<1, dim>> new_positions;
        std::deque<Tensor<1, dim>> new_velocities;
        std::deque<Tensor<1, dim>> new_accelerations;
        std::deque<double> new_angles;
        std::deque<double> new_angular_velocities;
        std::deque<double> new_angular_accelerations;

        if (positions) {
            std::cout << "# pre_time_loop: bdf1 ran earlier ??" << std::endl;
            // A BDF method of lower degree was probably run before this. We
            // therefore want to save the steps that has already been done.
            for (unsigned int k = positions.value().size() - 1; k >= 0; --k) {
                new_positions.push_front(positions.value()[k]);
                new_velocities.push_front(velocities.value()[k]);
                new_accelerations.push_front(accelerations.value()[k]);
                new_angles.push_front(angles.value()[k]);
                new_angular_velocities.push_front(
                        angular_velocities.value()[k]);
                new_angular_accelerations.push_front(
                        angular_accelerations.value()[k]);
            }
        } else {
            Tensor<1, dim> zero;
            new_positions.push_front(r0);
            new_velocities.push_front(zero);
            new_accelerations.push_front(zero);
            new_angles.push_front(0);
            new_angular_velocities.push_front(0);
            new_angular_accelerations.push_front(0);
            file << "0; 0;" << r0[0] << ";" << r0[1] << ";"
                 << "0;0;0;0;0;0;0" << std::endl;
        }
        positions = new_positions;
        velocities = new_velocities;
        accelerations = new_accelerations;
        angles = new_angles;
        angular_velocities = new_angular_velocities;
        angular_accelerations = new_angular_accelerations;
    }


    template<int dim>
    void FallingSphere<dim>::
    post_processing(unsigned int time_step) {
        std::cout << "Post processing " << std::endl;

        // Store all positions and accelerations.
        std::cout << positions.value().size() << " "
                  << velocities.value().size() << " "
                  << accelerations.value().size() << " "
                  << angles.value().size() << " "
                  << angular_velocities.value().size() << " "
                  << angular_accelerations.value().size() << std::endl;
        conservation_linear_momentum();

        conservation_angular_momentum();

        update_boundary_values();

        write_data(time_step);
    }


    template<int dim>
    void FallingSphere<dim>::
    conservation_linear_momentum() {
        Tensor<1, dim> surface_forces = this->compute_surface_forces();
        Tensor<1, dim> gravity;
        gravity[1] = -9.81;
        double sphere_volume = M_PI * pow(sphere_radius, 2);

        // Compute the acceleration of the sphere center
        double sphere_mass = sphere_density * sphere_volume;
        Tensor<1, dim> acceleration = gravity;
        Tensor<1, dim> sur_forces = surface_forces;
        acceleration += sur_forces;

        // TODO gravity seems to make sense, but not the surface forses?
        //  maybe the viscous forces are ruining thins. Maybe they explode when
        //  the flow reaches numbers somewhat above zero.
        std::cout << " - gravity = " << gravity << std::endl;
        std::cout << " - surface_forces = " << sur_forces << std::endl;
        std::cout << " - a = " << acceleration << std::endl;

        // Set the compute acceleration on the domain object
        // domain->set_acceleration(acceleration);

        // Compute the velocity and position of the next step.
        accelerations.value().push_front(acceleration);
        Tensor<1, dim> next_velocity =
                velocities.value()[0]
                + this->tau * acceleration;
        velocities.value().push_front(next_velocity);
        std::cout << " - v = " << next_velocity << std::endl;

        // Compute the new position to the sphere, based on the previous
        // position and the acceleration.
        Tensor<1, dim> next_position =
                positions.value()[0] + this->tau * next_velocity
                + pow(this->tau, 2) / 2 * acceleration;
        positions.value().push_front(next_position);
        std::cout << " - r = " << next_position << std::endl;
    }

    template<int dim>
    void FallingSphere<dim>::
    conservation_angular_momentum() {
        double surface_torque = compute_surface_torque();
        double sphere_volume = M_PI * pow(sphere_radius, 2);
        double sphere_mass = sphere_density * sphere_volume;

        double angular_acceleration =
                2 / (sphere_mass * pow(sphere_radius, 2)) * surface_torque;
        angular_accelerations.value().push_front(angular_acceleration);
        std::cout << " - alpha = " << angular_acceleration << std::endl;

        // Compute the angular velocity and angle of the next step.
        double next_angular_velocity =
                angular_velocities.value()[0]
                + this->tau * angular_acceleration;
        angular_velocities.value().push_front(next_angular_velocity);
        std::cout << " - omega = " << next_angular_velocity << std::endl;

        // TODO use a higher order method.
        double next_angle =
                angles.value()[0] + this->tau * next_angular_velocity
                + pow(this->tau, 2) / 2 * angular_acceleration;
        angles.value().push_front(next_angle);
        std::cout << " - theta = " << next_angle << std::endl;

    }

    template<int dim>
    void FallingSphere<dim>::
    update_boundary_values() {
        Tensor<1, dim> position = positions.value()[0];
        Tensor<1, dim> velocity = velocities.value()[0];
        double angular_velocity = angular_velocities.value()[0];

        boundary->set_sphere_center_position(position);
        boundary->set_sphere_center_velocity(velocity);
        boundary->set_sphere_angular_velocity(angular_velocity);

        // Update the position of the sphere center.
        domain->set_position(positions.value()[0]);
    }


    template<int dim>
    void FallingSphere<dim>::
    write_data(unsigned int time_step) {
        file << time_step << ";"
             << time_step * this->tau << ";"
             << positions.value()[0][0] << ";"
             << positions.value()[0][1] << ";"
             << velocities.value()[0][0] << ";"
             << velocities.value()[0][1] << ";"
             << accelerations.value()[0][0] << ";"
             << accelerations.value()[0][1] << ";"
             << angles.value()[0] << ";"
             << angular_velocities.value()[0] << ";"
             << angular_accelerations.value()[0] << std::endl;
    }


    template<int dim>
    double FallingSphere<dim>::
    compute_surface_torque() {
        NonMatching::RegionUpdateFlags region_update_flags;
        region_update_flags.inside = update_values | update_quadrature_points
                                     | update_JxW_values;
        region_update_flags.surface =
                update_values | update_JxW_values |
                update_gradients |
                update_quadrature_points |
                update_normal_vectors;

        NonMatching::FEValues<dim> cut_fe_values(this->mapping_collection,
                                                 this->fe_collection,
                                                 this->q_collection,
                                                 this->q_collection1D,
                                                 region_update_flags,
                                                 this->cut_mesh_classifier,
                                                 this->levelset_dof_handler,
                                                 this->levelset);

        double viscous_torque;
        double pressure_torque;
        for (const auto &cell : this->dof_handlers.front()->active_cell_iterators()) {
            const unsigned int n_dofs = cell->get_fe().dofs_per_cell;
            // This call will compute quadrature rules relevant for this cell
            // in the background.
            cut_fe_values.reinit(cell);

            // Retrieve an FEValues object with quadrature points
            // on the immersed surface.
            const boost::optional<const FEImmersedSurfaceValues<dim> &>
                    fe_values_surface = cut_fe_values.get_surface_fe_values();
            if (fe_values_surface) {
                integrate_surface_torque(*fe_values_surface,
                                         this->solutions.front(),
                                         viscous_torque,
                                         pressure_torque);
            }
        }
        double surface_torque = viscous_torque + pressure_torque;
        return surface_torque;
    }


    template<int dim>
    void FallingSphere<dim>::
    integrate_surface_torque(const FEValuesBase<dim> &fe_v,
                             Vector<double> solution,
                             double &viscous_torque,
                             double &pressure_torque) {

        const FEValuesExtractors::Vector v(0);
        const FEValuesExtractors::Scalar p(dim);

        // Vector of the values of the symmetric gradients for u on the
        // quadrature points.
        std::vector<Tensor<2, dim>> u_gradients(fe_v.n_quadrature_points);
        fe_v[v].get_function_gradients(solution, u_gradients);
        // Vector of the values of the pressure in the quadrature points.
        std::vector<double> p_values(fe_v.n_quadrature_points);
        fe_v[p].get_function_values(solution, p_values);

        std::vector<Point<dim>> q_points = fe_v.get_quadrature_points();

        Tensor<1, dim> v_forces;
        Tensor<1, dim> p_forces;
        Tensor<2, dim> I; // Identity matrix
        I[0][0] = 1;
        I[1][1] = 1;
        if (dim == 3) { I[2][2] = 1; }

        Tensor<1, dim> normal;
        Tensor<1, dim> center = positions.value()[0];
        // The vector pointing from the sphere center to the quadrature point.
        Tensor<1, dim> center_q_point;

        for (unsigned int q : fe_v.quadrature_point_indices()) {
            // We want the outward pointing normal of e.g. the sphere, so we
            // need to change the direction of the implemented normal, since
            // this normal points out of the domain.
            normal = -fe_v.normal_vector(q);

            // The viscous forces and pressure forces act on the submerged body.
            v_forces = this->nu * u_gradients[q] * normal;
            p_forces = -p_values[q] * I * normal;
            center_q_point = q_points[q] - center;

            viscous_torque +=
                    cross_product(center_q_point, v_forces) * fe_v.JxW(q);
            pressure_torque +=
                    cross_product(center_q_point, p_forces) * fe_v.JxW(q);
        }
    }

    template
    class FallingSphere<2>;

}
