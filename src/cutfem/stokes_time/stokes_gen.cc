#include <deal.II/base/data_out_base.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/non_matching/cut_mesh_classifier.h>
#include <deal.II/non_matching/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/optional.hpp>

#include <cmath>
#include <fstream>

#include "cutfem/geometry/SignedDistanceSphere.h"
#include "cutfem/nla/sparsity_pattern.h"
#include "cutfem/stabilization/jump_stabilization.h"

#include "../../utils/integration.h"
#include "../../utils/output.h"
#include "stokes_gen.h"


using namespace cutfem;

namespace TimeDependentStokesIE {

    template<int dim>
    StokesCylinder<dim>::StokesCylinder(const double radius,
                                        const double half_length,
                                        const unsigned int n_refines,
                                        const double nu,
                                        const double tau,
                                        const int element_order,
                                        const bool write_output,
                                        TensorFunction<1, dim> &rhs,
                                        TensorFunction<1, dim> &bdd_values,
                                        TensorFunction<1, dim> &analytic_vel,
                                        Function<dim> &analytic_pressure,
                                        const double sphere_radius,
                                        const double sphere_x_coord)
            : radius(radius), half_length(half_length), n_refines(n_refines),
              nu(nu), tau(tau),
              write_output(write_output), sphere_radius(sphere_radius),
              sphere_x_coord(sphere_x_coord),
              element_order(element_order),
              stokes_fe(FESystem<dim>(FE_Q<dim>(element_order + 1), dim),
                        1,
                        FE_Q<dim>(element_order),
                        1), fe_levelset(element_order),
              levelset_dof_handler(triangulation), dof_handler(triangulation),
              cut_mesh_classifier(triangulation, levelset_dof_handler,
                                  levelset) {
        h = 0;
        // Use no constraints when projecting.
        constraints.close();

        // Use Dirichlet boundary conditions everywhere.
        do_nothing_id = 10;

        rhs_function = &rhs;
        boundary_values = &bdd_values;
        analytical_velocity = &analytic_vel;
        analytical_pressure = &analytic_pressure;

        if (dim == 2) {
            this->center = Point<dim>(sphere_x_coord, 0);
        } else if (dim == 3) {
            this->center = Point<dim>(sphere_x_coord, 0, 0);
        }
    }

    template<int dim>
    void
    StokesCylinder<dim>::setup_quadrature() {
        const unsigned int quadOrder = 2 * element_order + 1;
        q_collection.push_back(QGauss<dim>(quadOrder));
        q_collection1D.push_back(QGauss<1>(quadOrder));
    }

    template<int dim>
    Error
    StokesCylinder<dim>::run(unsigned int steps) {
        make_grid();
        setup_quadrature();
        setup_level_set();
        cut_mesh_classifier.reclassify();
        distribute_dofs();
        initialize_matrices();
        old_solution.reinit(solution.size());
        assemble_system();

        double k = 0; // the time step index
        double time = 0;

        // Vector for the computed error for each time step.
        std::vector<Error> errors(steps);

        while (k < steps) {
            ++k;
            time += tau;
            std::cout << "\nTime Step = " << k
                      << ", time = " << time << std::endl;

            // TODO use advance_time instead?
            rhs_function->set_time(time);
            boundary_values->set_time(time);
            analytical_velocity->set_time(time);
            analytical_pressure->set_time(time);

            if (k == 1) {
                // Use the boundary_values as initial values. Interpolate the
                // boundary_values function into the finite element space.
                const unsigned int n_components_on_element = dim + 1;
                FEValuesExtractors::Vector velocities(0);
                VectorFunctionFromTensorFunction<dim> adapter(
                        *boundary_values,
                        velocities.first_vector_component,
                        n_components_on_element);
                VectorTools::interpolate(
                        dof_handler,
                        adapter,
                        solution,
                        fe_collection.component_mask(velocities));

                output_results(0);
                old_solution = solution;
            }

            // TODO nødvendig??
            solution.reinit(solution.size());
            rhs.reinit(solution.size());

            assemble_rhs();
            solve();
            if (write_output) {
                output_results(k);
            }
            old_solution = solution;
            errors[k - 1] = compute_error();
        }

        // TODO merk at det virker som om feilen er størst i starten for så å
        //  bli veldig mye mindre etter som t vokser. Er dette feil, eller er
        //  det fordi løsningen faktisk blir så mye mindre? (i p-L2 feil)
        for (Error err : errors) {
            std::cout << "u-l2= " << err.l2_error_u
                      << "    u-h1= " << err.h1_error_u << std::endl;
        }

        return compute_time_error(errors);
    }

    template<int dim>
    void
    StokesCylinder<dim>::make_grid() {
        std::cout << "Creating triangulation" << std::endl;

        GridGenerator::cylinder(triangulation, radius, half_length);
        GridTools::remove_anisotropy(triangulation, 1.618, 5);
        triangulation.refine_global(n_refines);

        mapping_collection.push_back(MappingCartesian<dim>());

        // Save the cell-size, we need it in the Nitsche term.
        typename Triangulation<dim>::active_cell_iterator cell =
                triangulation.begin_active();
        h = std::pow(cell->measure(), 1.0 / dim);
    }

    template<int dim>
    void
    StokesCylinder<dim>::setup_level_set() {
        std::cout << "Setting up level set" << std::endl;

        // The level set function lives on the whole background mesh.
        levelset_dof_handler.distribute_dofs(fe_levelset);
        printf("leveset dofs: %d\n", levelset_dof_handler.n_dofs());
        levelset.reinit(levelset_dof_handler.n_dofs());

        // Project the geometry onto the mesh.
        cutfem::geometry::SignedDistanceSphere<dim> signed_distance_sphere(
                sphere_radius, center, -1);
        VectorTools::project(levelset_dof_handler,
                             constraints,
                             QGauss<dim>(2 * element_order + 1),
                             signed_distance_sphere,
                             levelset);
    }

    template<int dim>
    void
    StokesCylinder<dim>::distribute_dofs() {
        std::cout << "Distributing dofs" << std::endl;

        // We want to types of elements on the mesh
        // Lagrange elements and elements that are constant zero..
        fe_collection.push_back(stokes_fe);
        fe_collection.push_back(FESystem<dim>(
                FESystem<dim>(FE_Nothing<dim>(), dim), 1, FE_Nothing<dim>(),
                1));

        // Set outside finite elements to stokes_fe, and inside to FE_nothing
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            if (LocationToLevelSet::OUTSIDE ==
                cut_mesh_classifier.location_to_level_set(cell)) {
                // 1 is FE_nothing
                cell->set_active_fe_index(1);
            } else {
                // 0 is stokes_fe
                cell->set_active_fe_index(0);
            }
        }
        dof_handler.distribute_dofs(fe_collection);
    }

    template<int dim>
    void
    StokesCylinder<dim>::initialize_matrices() {
        std::cout << "Initialize marices" << std::endl;
        solution.reinit(dof_handler.n_dofs());
        rhs.reinit(dof_handler.n_dofs());

        cutfem::nla::make_sparsity_pattern_for_stabilized(dof_handler,
                                                          sparsity_pattern);
        stiffness_matrix.reinit(sparsity_pattern);
    }

    template<int dim>
    void
    StokesCylinder<dim>::assemble_system() {
        std::cout << "Assembling" << std::endl;

        // Use a helper object to compute the stabilisation for both the velocity
        // and the pressure component.
        const FEValuesExtractors::Vector velocities(0);
        stabilization::JumpStabilization<dim, FEValuesExtractors::Vector>
                velocity_stab(dof_handler,
                              mapping_collection,
                              cut_mesh_classifier,
                              constraints);
        velocity_stab.set_function_describing_faces_to_stabilize(
                stabilization::inside_stabilization);
        velocity_stab.set_weight_function(stabilization::taylor_weights);
        velocity_stab.set_extractor(velocities);

        const FEValuesExtractors::Scalar pressure(dim);
        stabilization::JumpStabilization<dim, FEValuesExtractors::Scalar>
                pressure_stab(dof_handler,
                              mapping_collection,
                              cut_mesh_classifier,
                              constraints);
        pressure_stab.set_function_describing_faces_to_stabilize(
                stabilization::inside_stabilization);
        pressure_stab.set_weight_function(stabilization::taylor_weights);
        pressure_stab.set_extractor(pressure);

        double beta_0 = 0.1;
        double gamma_A = beta_0 * element_order * element_order;
        double gamma_M = beta_0 * element_order * element_order;

        NonMatching::RegionUpdateFlags region_update_flags;
        region_update_flags.inside = update_values | update_JxW_values |
                                     update_gradients |
                                     update_quadrature_points;
        region_update_flags.surface = update_values | update_JxW_values |
                                      update_gradients |
                                      update_quadrature_points |
                                      update_normal_vectors;

        NonMatching::FEValues<dim> cut_fe_values(mapping_collection,
                                                 fe_collection,
                                                 q_collection,
                                                 q_collection1D,
                                                 region_update_flags,
                                                 cut_mesh_classifier,
                                                 levelset_dof_handler,
                                                 levelset);

        // Quadrature for the faces of the cells on the outer boundary
        QGauss<dim - 1> face_quadrature_formula(stokes_fe.degree + 1);
        FEFaceValues<dim> fe_face_values(stokes_fe,
                                         face_quadrature_formula,
                                         update_values | update_gradients |
                                         update_quadrature_points |
                                         update_normal_vectors |
                                         update_JxW_values);

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            const unsigned int n_dofs = cell->get_fe().dofs_per_cell;
            std::vector<types::global_dof_index> loc2glb(n_dofs);
            cell->get_dof_indices(loc2glb);

            // This call will compute quadrature rules relevant for this cell
            // in the background.
            cut_fe_values.reinit(cell);

            // Retrieve an FEValues object with quadrature points
            // over the full cell.
            const boost::optional<const FEValues<dim> &> fe_values_bulk =
                    cut_fe_values.get_inside_fe_values();

            if (fe_values_bulk)
                assemble_local_over_cell(*fe_values_bulk, loc2glb);

            // Loop through all faces that constitutes the outer boundary of the
            // domain.
            for (const auto &face : cell->face_iterators()) {
                if (face->at_boundary() &&
                    face->boundary_id() != do_nothing_id) {
                    fe_face_values.reinit(cell, face);
                    assemble_local_over_surface(fe_face_values, loc2glb);
                }
            }

            // Retrieve an FEValues object with quadrature points
            // on the immersed surface.
            const boost::optional<const FEImmersedSurfaceValues<dim> &>
                    fe_values_surface = cut_fe_values.get_surface_fe_values();

            if (fe_values_surface)
                assemble_local_over_surface(*fe_values_surface, loc2glb);

            // Compute and add the velocity stabilization.
            velocity_stab.compute_stabilization(cell);
            velocity_stab.add_stabilization_to_matrix(
                    gamma_M * delta + gamma_A * tau * nu / (h * h),
                    stiffness_matrix);
            // Compute and add the pressure stabilisation.
            pressure_stab.compute_stabilization(cell);
            pressure_stab.add_stabilization_to_matrix(-gamma_A,
                                                      stiffness_matrix);
            // TODO et stabilierings ledd til for trykket også?
        }
    }

    template<int dim>
    void
    StokesCylinder<dim>::assemble_local_over_cell(
            const FEValues<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {

        // Matrix and vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // Vector for values of the RightHandSide for all quadrature points on a cell.
        std::vector<Tensor<1, dim>> rhs_values(fe_values.n_quadrature_points,
                                               Tensor<1, dim>());
        rhs_function->value_list(fe_values.get_quadrature_points(), rhs_values);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        // Calculate often used terms in the beginning of each cell-loop
        std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
        std::vector<double> div_phi_u(dofs_per_cell);
        std::vector<Tensor<1, dim>> phi_u(dofs_per_cell, Tensor<1, dim>());
        std::vector<double> phi_p(dofs_per_cell);

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
            for (const unsigned int k : fe_values.dof_indices()) {
                grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                div_phi_u[k] = fe_values[velocities].divergence(k, q);
                phi_u[k] = fe_values[velocities].value(k, q);
                phi_p[k] = fe_values[pressure].value(k, q);
            }

            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    local_matrix(i, j) +=
                            (delta * phi_u[j] * phi_u[i]  // (u, v)
                             +
                             (nu * scalar_product(grad_phi_u[j],
                                                  grad_phi_u[i]) // (grad u, grad v)
                              - (div_phi_u[i] * phi_p[j])   // -(div v, p)
                              - (div_phi_u[j] * phi_p[i])   // -(div u, q)
                             ) * tau) *
                            fe_values.JxW(q); // dx
                }
                // NB: rhs is assembled in assemble_rhs().
            }
        }
        stiffness_matrix.add(loc2glb, local_matrix);
    }


    template<int dim>
    void
    StokesCylinder<dim>::assemble_local_over_surface(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {
        // Matrix and vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // Evaluate the boundary function for all quadrature points on this face.
        std::vector<Tensor<1, dim>> bdd_values(fe_values.n_quadrature_points,
                                               Tensor<1, dim>());
        boundary_values->value_list(fe_values.get_quadrature_points(),
                                    bdd_values);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        // Calculate often used terms in the beginning of each cell-loop
        std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
        std::vector<double> div_phi_u(dofs_per_cell);
        std::vector<Tensor<1, dim>> phi_u(dofs_per_cell, Tensor<1, dim>());
        std::vector<double> phi_p(dofs_per_cell);

        // TODO denne skal vel avhenge av element_order?
        double mu = 50 / h; // Nitsche penalty parameter
        Tensor<1, dim> normal;

        for (unsigned int q : fe_values.quadrature_point_indices()) {
            normal = fe_values.normal_vector(q);

            for (const unsigned int k : fe_values.dof_indices()) {
                grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                div_phi_u[k] = fe_values[velocities].divergence(k, q);
                phi_u[k] = fe_values[velocities].value(k, q);
                phi_p[k] = fe_values[pressure].value(k, q);
            }

            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    local_matrix(i, j) +=
                            (-nu * (grad_phi_u[j] * normal) *
                             phi_u[i]  // -(grad u * n, v)
                             -
                             (grad_phi_u[i] * normal) *
                             phi_u[j] // -(grad v * n, u) [Nitsche]
                             + mu * (phi_u[j] * phi_u[i]) // mu (u, v) [Nitsche]
                             + (normal * phi_u[i]) *
                               phi_p[j]                  // (n * v, p) [from ∇p]
                             + (normal * phi_u[j]) *
                               phi_p[i]                  // (q*n, u) [Nitsche]
                            ) * tau *  // Multiply all terms with the time step
                            fe_values.JxW(q); // ds
                }
                // NB: rhs is assembled in assemble_rhs().
            }
        }
        stiffness_matrix.add(loc2glb, local_matrix);
    }


    template<int dim>
    void StokesCylinder<dim>::
    assemble_rhs() {
        std::cout << "Assembling rhs" << std::endl;

        NonMatching::RegionUpdateFlags region_update_flags;
        region_update_flags.inside = update_values | update_quadrature_points
                                     |
                                     update_JxW_values; //  | update_gradients;
        region_update_flags.surface =
                update_values | update_JxW_values |
                update_gradients |
                update_quadrature_points |
                update_normal_vectors;

        NonMatching::FEValues<dim> cut_fe_values(mapping_collection,
                                                 fe_collection,
                                                 q_collection,
                                                 q_collection1D,
                                                 region_update_flags,
                                                 cut_mesh_classifier,
                                                 levelset_dof_handler,
                                                 levelset);

        // Quadrature for the faces of the cells on the outer boundary
        QGauss<dim - 1> face_quadrature_formula(stokes_fe.degree + 1);
        FEFaceValues<dim> fe_face_values(stokes_fe,
                                         face_quadrature_formula,
                                         update_values | update_gradients |
                                         update_quadrature_points |
                                         update_normal_vectors |
                                         update_JxW_values);

        // TODO setter dette alle elementene i rhs til 0?
        // rhs = 0;

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            const unsigned int n_dofs = cell->get_fe().dofs_per_cell;
            std::vector<types::global_dof_index> loc2glb(n_dofs);
            cell->get_dof_indices(loc2glb);

            // This call will compute quadrature rules relevant for this cell
            // in the background.
            cut_fe_values.reinit(cell);

            // Retrieve an FEValues object with quadrature points
            // over the full cell.
            const boost::optional<const FEValues<dim> &> fe_values_bulk =
                    cut_fe_values.get_inside_fe_values();

            if (fe_values_bulk) {
                assemble_rhs_local_over_cell(*fe_values_bulk, loc2glb);
            }

            // Loop through all faces that constitutes the outer boundary of the
            // domain.
            for (const auto &face : cell->face_iterators()) {
                if (face->at_boundary() &&
                    face->boundary_id() != do_nothing_id) {
                    fe_face_values.reinit(cell, face);
                    assemble_rhs_local_over_surface(fe_face_values, loc2glb);
                }
            }

            // Retrieve an FEValues object with quadrature points
            // on the immersed surface.
            const boost::optional<const FEImmersedSurfaceValues<dim> &>
                    fe_values_surface = cut_fe_values.get_surface_fe_values();

            if (fe_values_surface) {
                assemble_rhs_local_over_surface(*fe_values_surface, loc2glb);
            }
        }
    }


    template<int dim>
    void StokesCylinder<dim>::
    assemble_rhs_local_over_cell(
            const FEValues<dim> &fe_v,
            const std::vector<types::global_dof_index> &loc2glb) {

        // Matrix and vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_v.get_fe().dofs_per_cell;
        // FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // Vector for values of the RightHandSide for all quadrature points on a cell.
        std::vector<Tensor<1, dim>> rhs_values(fe_v.n_quadrature_points,
                                               Tensor<1, dim>());
        rhs_function->value_list(fe_v.get_quadrature_points(), rhs_values);

        // Get the values from the solution in the last time step.
        const FEValuesExtractors::Vector v(0);
        std::vector<Tensor<1, dim>> u_solution_values(fe_v.n_quadrature_points);
        fe_v[v].get_function_values(old_solution, u_solution_values);
        Tensor<1, dim> phi_u;

        for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q) {
            for (const unsigned int i : fe_v.dof_indices()) {
                // RHS
                phi_u = fe_v[v].value(i, q);
                local_rhs(i) += (tau * rhs_values[q] * phi_u // (f, v)
                                 +
                                 u_solution_values[q] * phi_u
                                ) * fe_v.JxW(q);      // dx
            }
        }
        rhs.add(loc2glb, local_rhs);
    }


    template<int dim>
    void StokesCylinder<dim>::
    assemble_rhs_local_over_surface(
            const FEValuesBase<dim> &fe_v,
            const std::vector<types::global_dof_index> &loc2glb) {
        // Matrix and vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_v.get_fe().dofs_per_cell;
        // FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // Evaluate the boundary function for all quadrature points on this face.
        std::vector<Tensor<1, dim>> bdd_values(fe_v.n_quadrature_points,
                                               Tensor<1, dim>());
        boundary_values->value_list(fe_v.get_quadrature_points(), bdd_values);

        const FEValuesExtractors::Vector v(0);
        const FEValuesExtractors::Scalar p(dim);

        // TODO denne skal vel avhenge av element_order?
        double mu = 50 / h; // Nitsche penalty parameter
        Tensor<1, dim> normal;

        for (unsigned int q : fe_v.quadrature_point_indices()) {
            normal = fe_v.normal_vector(q);

            for (const unsigned int i : fe_v.dof_indices()) {
                // These terms comes from Nitsches method.
                Tensor<1, dim> prod_r =
                        mu * fe_v[v].value(i, q) -
                        fe_v[v].gradient(i, q) * normal +
                        fe_v[p].value(i, q) * normal;

                local_rhs(i) +=
                        tau * prod_r *
                        bdd_values[q] // (g, mu v - n grad v + q * n)
                        * fe_v.JxW(q);    // ds
            }
        }
        rhs.add(loc2glb, local_rhs);
    }

    template<int dim>
    void
    StokesCylinder<dim>::solve() {
        std::cout << "Solving system" << std::endl;
        SparseDirectUMFPACK inverse;
        inverse.initialize(stiffness_matrix);
        inverse.vmult(solution, rhs);
    }

    template<int dim>
    void StokesCylinder<dim>::
    output_results(int time_step) const {
        std::cout << "Output results" << std::endl;

        std::ofstream output("solution-d" + std::to_string(dim)
                             + "o" + std::to_string(element_order)
                             + "r" + std::to_string(n_refines)
                             + "t" + std::to_string(time_step) + ".vtk");
        Utils::writeNumericalSolution(dof_handler, solution, output);

        std::ofstream output_ex("analytical-d" + std::to_string(dim)
                                + "o" + std::to_string(element_order)
                                + "r" + std::to_string(n_refines)
                                + "t" + std::to_string(time_step) + ".vtk");
        std::ofstream file_diff("diff-d" + std::to_string(dim)
                                + "o" + std::to_string(element_order)
                                + "r" + std::to_string(n_refines)
                                + "t" + std::to_string(time_step) + ".vtk");
        Utils::writeAnalyticalSolutionAndDiff(dof_handler,
                                              fe_collection,
                                              solution,
                                              *analytical_velocity,
                                              *analytical_pressure,
                                              output_ex,
                                              file_diff);

        // Output levelset function.
        DataOut<dim, DoFHandler<dim>> data_out_levelset;
        data_out_levelset.attach_dof_handler(levelset_dof_handler);
        data_out_levelset.add_data_vector(levelset, "levelset");
        data_out_levelset.build_patches();
        std::ofstream output_ls("levelset-d" + std::to_string(dim)
                                + "o" + std::to_string(element_order)
                                + "r" + std::to_string(n_refines)
                                + "t" + std::to_string(time_step) + ".vtk");
        data_out_levelset.write_vtk(output_ls);
    }


    template<int dim>
    Error StokesCylinder<dim>::
    compute_error() {
        // TODO move to integration.h
        std::cout << "Compute error" << std::endl;
        std::cout << "u-time=" << analytical_velocity->get_time() << " p-time="
                  << analytical_pressure->get_time() << std::endl;

        NonMatching::RegionUpdateFlags region_update_flags;
        region_update_flags.inside = update_values | update_JxW_values |
                                     update_gradients |
                                     update_quadrature_points;
        region_update_flags.surface = update_values | update_JxW_values |
                                      update_gradients |
                                      update_quadrature_points |
                                      update_normal_vectors;

        NonMatching::FEValues<dim> cut_fe_values(mapping_collection,
                                                 fe_collection,
                                                 q_collection,
                                                 q_collection1D,
                                                 region_update_flags,
                                                 cut_mesh_classifier,
                                                 levelset_dof_handler,
                                                 levelset);

        // Compute the mean of the numerical and the exact pressure over the
        // domain, to subtract it before computing the error.
        double mean_num_pressure = 0;
        double mean_ext_pressure = 0;
        Utils::compute_mean_pressure(dof_handler,
                                     cut_fe_values,
                                     solution,
                                     *analytical_pressure,
                                     mean_num_pressure,
                                     mean_ext_pressure);

        double l2_error_integral_u = 0;
        double h1_error_integral_u = 0;
        double l2_error_integral_p = 0;
        double h1_error_integral_p = 0;

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            cut_fe_values.reinit(cell);

            const boost::optional<const FEValues<dim> &> fe_values_inside =
                    cut_fe_values.get_inside_fe_values();

            if (fe_values_inside) {
                integrate_cell(*fe_values_inside, l2_error_integral_u,
                               h1_error_integral_u, l2_error_integral_p,
                               h1_error_integral_p, mean_num_pressure,
                               mean_ext_pressure);
            }
        }

        Error error;
        error.mesh_size = h;
        error.l2_error_u = pow(l2_error_integral_u, 0.5);
        error.h1_error_u = pow(l2_error_integral_u + h1_error_integral_u, 0.5);
        error.h1_semi_u = pow(h1_error_integral_u, 0.5);
        error.l2_error_p = pow(l2_error_integral_p, 0.5);
        error.h1_error_p = pow(l2_error_integral_p + h1_error_integral_p, 0.5);
        error.h1_semi_p = pow(h1_error_integral_p, 0.5);
        return error;
    }


    template<int dim>
    void StokesCylinder<dim>::
    integrate_cell(const FEValues<dim> &fe_v,
                   double &l2_error_integral_u,
                   double &h1_error_integral_u,
                   double &l2_error_integral_p,
                   double &h1_error_integral_p,
                   const double &mean_numerical_pressure,
                   const double &mean_exact_pressure) const {

        const FEValuesExtractors::Vector v(0);
        const FEValuesExtractors::Scalar p(dim);

        std::vector<Tensor<1, dim>> u_solution_values(fe_v.n_quadrature_points);
        std::vector<Tensor<2, dim>> u_solution_gradients(
                fe_v.n_quadrature_points);
        std::vector<double> p_solution_values(fe_v.n_quadrature_points);
        std::vector<Tensor<1, dim>> p_solution_gradients(
                fe_v.n_quadrature_points);

        fe_v[v].get_function_values(solution, u_solution_values);
        fe_v[v].get_function_gradients(solution, u_solution_gradients);
        fe_v[p].get_function_values(solution, p_solution_values);
        fe_v[p].get_function_gradients(solution, p_solution_gradients);

        // Exact solution: velocity and pressure
        std::vector<Tensor<1, dim>> u_exact_solution(
                fe_v.n_quadrature_points,
                Tensor<1, dim>());
        std::vector<double> p_exact_solution(fe_v.n_quadrature_points);
        analytical_velocity->value_list(fe_v.get_quadrature_points(),
                                        u_exact_solution);
        analytical_pressure->value_list(fe_v.get_quadrature_points(),
                                        p_exact_solution);

        // Exact gradients: velocity and pressure
        std::vector<Tensor<2, dim>> u_exact_gradients(fe_v.n_quadrature_points,
                                                      Tensor<2, dim>());
        std::vector<Tensor<1, dim>> p_exact_gradients(fe_v.n_quadrature_points,
                                                      Tensor<1, dim>());
        analytical_velocity->gradient_list(fe_v.get_quadrature_points(),
                                           u_exact_gradients);
        analytical_pressure->gradient_list(fe_v.get_quadrature_points(),
                                           p_exact_gradients);

        for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q) {
            // Integrate the square difference between exact and numeric solution
            // for function values and gradients (both pressure and velocity).
            Tensor<1, dim> diff_u = u_exact_solution[q] - u_solution_values[q];
            double diff_p = (p_exact_solution[q] - mean_exact_pressure) -
                            (p_solution_values[q] - mean_numerical_pressure);

            Tensor<2, dim> diff_u_gradient =
                    u_exact_gradients[q] - u_solution_gradients[q];
            Tensor<1, dim> diff_p_gradient =
                    p_exact_gradients[q] - p_solution_gradients[q];

            l2_error_integral_u += diff_u * diff_u * fe_v.JxW(q);
            l2_error_integral_p += diff_p * diff_p * fe_v.JxW(q);

            h1_error_integral_u +=
                    scalar_product(diff_u_gradient, diff_u_gradient) *
                    fe_v.JxW(q);
            h1_error_integral_p +=
                    diff_p_gradient * diff_p_gradient * fe_v.JxW(q);
        }
    }


    /**
     * Compute the L2 and H1 error based on the computed error from each time
     * step.
     *
     * Compute the square root of the sum of the squared errors from each time
     * steps, weighted by the time step length tau for each term in the sum.
     */
    template<int dim>
    Error StokesCylinder<dim>::
    compute_time_error(std::vector<Error> errors) {
        double l2_error_integral_u = 0;
        double h1_error_integral_u = 0;
        double l2_error_integral_p = 0;
        double h1_error_integral_p = 0;

        for (Error error : errors) {
            l2_error_integral_u += tau * pow(error.l2_error_u, 2);
            h1_error_integral_u += tau * pow(error.h1_semi_u, 2);
            l2_error_integral_p += tau * pow(error.l2_error_p, 2);
            h1_error_integral_p += tau * pow(error.h1_semi_p, 2);
        }

        Error error;
        error.mesh_size = errors[0].mesh_size;
        error.l2_error_u = pow(l2_error_integral_u, 0.5);
        error.h1_error_u = pow(l2_error_integral_u + h1_error_integral_u, 0.5);
        error.h1_semi_u = pow(h1_error_integral_u, 0.5);
        error.l2_error_p = pow(l2_error_integral_p, 0.5);
        error.h1_error_p = pow(l2_error_integral_p + h1_error_integral_p, 0.5);
        error.h1_semi_p = pow(h1_error_integral_p, 0.5);
        return error;
    }


    template<int dim>
    void StokesCylinder<dim>::
    write_header_to_file(std::ofstream &file) {
        file << "h, \\|u\\|_{L^2}, \\|u\\|_{H^1}, |u|_{H^1}, "
                "\\|p\\|_{L^2}, \\|p\\|_{H^1}, |p|_{H^1}" << std::endl;
    }


    template<int dim>
    void StokesCylinder<dim>::
    write_error_to_file(Error &error, std::ofstream &file) {
        file << error.mesh_size << ","
             << error.l2_error_u << ","
             << error.h1_error_u << ","
             << error.h1_semi_u << ","
             << error.l2_error_p << ","
             << error.h1_error_p << ","
             << error.h1_semi_p << std::endl;
    }


    template
    class StokesCylinder<2>;

    template
    class StokesCylinder<3>;

} // namespace GeneralizedStokes
