#include <deal.II/non_matching/fe_values.h>

#include "ex_falling_sphere.h"

using namespace dealii;
using namespace cutfem;

namespace examples::cut::StokesEquation::ex2 {
    using NonMatching::LocationToLevelSet;

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
    BoundaryValues<dim>::BoundaryValues(const double sphere_radius,
                                        const double half_length,
                                        const double radius)
            : TensorFunction<1, dim>(), sphere_radius(sphere_radius),
              half_length(half_length), radius(radius) {}

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
        return ball_velocity;
    }

    template<int dim>
    void BoundaryValues<dim>::
    set_ball_velocity(Tensor<1, dim> value) {
        ball_velocity = value;
    }


    template<int dim>
    MovingDomain<dim>::MovingDomain(const double sphere_radius,
                                    const double half_length,
                                    const double radius,
                                    const double tau)
            : sphere_radius(sphere_radius), half_length(half_length),
              radius(radius), tau(tau) {}

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
    update_position() {
        last_position = new_position;
        last_velocity = new_velocity;

        new_velocity = last_velocity + acceleration * tau;
        new_position = last_position
                       + last_velocity * tau
                       + 0.5 * acceleration * tau * tau;
        std::cout << " - new v = " << new_velocity << std::endl;
        std::cout << " - new r = " << new_position << std::endl;
    }

    template<int dim>
    void MovingDomain<dim>::
    set_acceleration(Tensor<1, dim> value) {
        acceleration = value;
    }

    template<int dim>
    double MovingDomain<dim>::
    get_volume() {
        return numbers::PI * pow(sphere_radius, 2);
    }

    template<int dim>
    Tensor<1, dim> MovingDomain<dim>::
    get_velocity() { return new_velocity; }


    template<int dim>
    FallingSphereStokes<dim>::
    FallingSphereStokes(const double nu,
                        const double tau,
                        const double radius,
                        const double half_length,
                        const unsigned int n_refines,
                        const int element_order,
                        const bool write_output,
                        TensorFunction<1, dim> &rhs,
                        TensorFunction<1, dim> &bdd_values,
                        TensorFunction<1, dim> &analytic_vel,
                        Function<dim> &analytic_pressure,
                        LevelSet<dim> &levelset_func,
                        const double density_ratio,
                        const int do_nothing_id)
            : StokesEqn<dim>(nu, tau, radius, half_length, n_refines,
                             element_order, write_output, rhs, bdd_values,
                             analytic_vel, analytic_pressure,
                             levelset_func, do_nothing_id, true, false),
              density_ratio(density_ratio) {
        file = std::ofstream("ex2_falling_sphere_data.csv");
        // TODO output physical quantities to this file.

    }

    template<int dim>
    void FallingSphereStokes<dim>::
    post_processing() {
        std::cout << "Post processing " << std::endl;

        auto *domain_func = dynamic_cast<MovingDomain<dim> *>(this->levelset_function);
        auto *boundary_values = dynamic_cast<BoundaryValues<dim> *>(this->boundary_values);

        Tensor<1, dim> surface_forces = compute_surface_forces();
        Tensor<1, dim> gravity;
        gravity[1] = -9.81;
        double sphere_volume = domain_func->get_volume();

        // Compute the acceleration
        Tensor<1, dim> acceleration = gravity;
        Tensor<1, dim> sur_forces = - density_ratio * surface_forces /
                                    sphere_volume; // TODO check sign: is the norm pointing the wrong way?
        acceleration += sur_forces;

        // TODO gravity seems to make sense, but not the surface forses?
        //  maybe the viscous forces are ruining thins. Maybe they explode when
        //  the flow reaches numbers somewhat above zero.
        std::cout << " - gravity = " << gravity << std::endl;
        std::cout << " - surface_forces = " << sur_forces << std::endl;
        std::cout << " - a = " << acceleration << std::endl;

        // Set the compute acceleration on the domain object
        domain_func->set_acceleration(acceleration);
        boundary_values->set_ball_velocity(domain_func->get_velocity());

        // Compute the new position to the sphere, based on the previous
        // position and the acceleration.
        domain_func->update_position();
    }

    template<int dim>
    Tensor<1, dim> FallingSphereStokes<dim>::
    compute_surface_forces() {
        std::cout << " - Compute the surface forces on the body." << std::endl;

        // Compute integral of the
        // - symmetric gradient over the boundary
        // - the pressure over the boundary

        NonMatching::RegionUpdateFlags region_update_flags;
        region_update_flags.inside = update_values | update_quadrature_points
                                     |
                                     update_JxW_values; //  | update_gradients;
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

        Tensor<1, dim> viscous_forces;
        Tensor<1, dim> pressure_forces;
        for (const auto &cell : this->dof_handlers.front()->active_cell_iterators()) {
            const unsigned int n_dofs = cell->get_fe().dofs_per_cell;
            std::vector<types::global_dof_index> loc2glb(n_dofs);
            cell->get_dof_indices(loc2glb);

            // This call will compute quadrature rules relevant for this cell
            // in the background.
            cut_fe_values.reinit(cell);

            // Retrieve an FEValues object with quadrature points
            // on the immersed surface.
            const boost::optional<const FEImmersedSurfaceValues<dim> &>
                    fe_values_surface = cut_fe_values.get_surface_fe_values();
            if (fe_values_surface) {
                integrate_surface_forces(*fe_values_surface,
                                         this->solutions.front(),
                                         viscous_forces,
                                         pressure_forces);
            }
        }
        Tensor<1, dim> surface_forces = viscous_forces + pressure_forces;
        std::cout << " - Viscous forces = " << viscous_forces << std::endl;
        std::cout << " - Pressure forces = " << pressure_forces << std::endl;
        return surface_forces;
    }


    template<int dim>
    void FallingSphereStokes<dim>::
    integrate_surface_forces(const FEValuesBase<dim> &fe_v,
                             Vector<double> solution,
                             Tensor<1, dim> &viscous_forces,
                             Tensor<1, dim> &pressure_forces) {

        const FEValuesExtractors::Vector v(0);
        const FEValuesExtractors::Scalar p(dim);

        // Vector of the values of the symmetric gradients for u on the
        // quadrature points.
        std::vector<Tensor<2, dim>> u_gradients(fe_v.n_quadrature_points);
        fe_v[v].get_function_gradients(solution, u_gradients);
        // Vector of the values of the pressure in the quadrature points.
        std::vector<double> p_values(fe_v.n_quadrature_points);
        fe_v[p].get_function_values(solution, p_values);

        Tensor<1, dim> v_forces;
        Tensor<1, dim> p_forces;
        Tensor<2, dim> I; // Identity matrix
        I[0][0] = 1;
        I[1][1] = 1;
        if (dim == 3) { I[2][2] = 1; }

        Tensor<1, dim> normal;
        for (unsigned int q : fe_v.quadrature_point_indices()) {
            normal = fe_v.normal_vector(q);
            // Get the transposed u-gradient, for the symmetric gradient.
            Tensor<2, dim> transposed = transpose(u_gradients[q]);

            // The viscous forces and pressure forces act on the submerged body.
            v_forces = this->nu * (u_gradients[q] + transposed) * normal;
            p_forces = -p_values[q] * I * normal;
            viscous_forces += v_forces * fe_v.JxW(q);
            pressure_forces += p_forces * fe_v.JxW(q);
        }
    }

    template
    class RightHandSide<2>;

    template
    class BoundaryValues<2>;

    template
    class MovingDomain<2>;

    template
    class FallingSphereStokes<2>;

}
