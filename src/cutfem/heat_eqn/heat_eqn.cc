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

#include "cutfem/nla/sparsity_pattern.h"
#include "cutfem/stabilization/jump_stabilization.h"

#include "heat_eqn.h"


using namespace cutfem;


namespace examples::cut::HeatEquation {


    template<int dim>
    HeatEqn<dim>::HeatEqn(const double nu,
                          const double tau,
                          const double radius,
                          const double half_length,
                          const unsigned int n_refines,
                          const int element_order,
                          const bool write_output,
                          Function<dim> &rhs,
                          Function<dim> &bdd_values,
                          Function<dim> &analytical_soln,
                          Function<dim> &domain_func,
                          const bool stabilized)
            : nu(nu), tau(tau),
              radius(radius), half_length(half_length), n_refines(n_refines),
              write_output(write_output), stabilized(stabilized),
              element_order(element_order),
              fe(element_order), fe_levelset(element_order),
              levelset_dof_handler(triangulation), dof_handler(triangulation),
              cut_mesh_classifier(triangulation, levelset_dof_handler,
                                  levelset) {
        // Use no constraints when projecting.
        constraints.close();

        rhs_function = &rhs;
        boundary_values = &bdd_values;
        analytical_solution = &analytical_soln;
        domain_function = &domain_func;
    }

    template<int dim>
    void
    HeatEqn<dim>::setup_quadrature() {
        // TODO
        const unsigned int quadOrder = 2 * element_order + 1;
        q_collection.push_back(QGauss<dim>(quadOrder));
        q_collection1D.push_back(QGauss<1>(quadOrder));
    }

    template<int dim>
    Error
    HeatEqn<dim>::run(unsigned int bdf_type, unsigned int steps) {
        // TODO imlement bdf2

        make_grid();
        setup_quadrature();
        setup_level_set();
        cut_mesh_classifier.reclassify();
        distribute_dofs();
        initialize_matrices();

        std::vector<Error> errors(steps + 1);
        interpolate_first_steps(bdf_type, errors);
        set_bdf_coefficients(bdf_type);
        assemble_matrix();

        // TODO BDF-2: if u1 is provided; compute the error that step.

        for (unsigned int k = bdf_type; k <= steps; ++k) {
            std::cout << "k = " << std::to_string(k) << std::endl;
            std::cout << "=========================" << std::endl;

            rhs_function->set_time(k * tau);
            boundary_values->set_time(k * tau);
            analytical_solution->set_time(k * tau);

            assemble_rhs();
            solve();
            errors[k] = compute_error();

            std::string suffix = "-" + std::to_string(k);
            if (write_output) {
                output_results(suffix);
            }

            for (unsigned long i = 1; i < solutions.size(); ++i) {
                solutions[i - 1] = solutions[i];
            }
            solutions[solutions.size() - 1] = solution;
        }

        // compute_condition_number();
        return compute_time_error(errors);
    }


    template<int dim>
    void HeatEqn<dim>::
    set_bdf_coefficients(unsigned int bdf_type) {
        bdf_coeffs = std::vector<double>(bdf_type + 1);

        if (bdf_type == 1) {
            bdf_coeffs[0] = 1;
            bdf_coeffs[1] = -1;
        } else {
            throw std::invalid_argument("Only BDF-1 is implemented for now.");
        }
    }


    template<int dim>
    void HeatEqn<dim>::
    interpolate_first_steps(unsigned int bdf_type, std::vector<Error> &errors) {
        solutions = std::vector<Vector<double>>(bdf_type);

        std::cout << "Fix interpolation!" << std::endl;

        for (unsigned int i = 0; i < bdf_type; ++i) {
            // Interpolate step i (step u1 will be overwritten by bdf2 if
            // u1 is provided).
            analytical_solution->set_time(i * tau);
            VectorTools::interpolate(dof_handler, *analytical_solution,
                                     solution);
            solutions[i].reinit(solution.size());

            // Compute the error of the interpolated step.
            errors[i] = compute_error();
            std::string suffix = "-" + std::to_string(i) + "-inter";
            output_results(suffix);
            solutions[i] = solution;
        }
    }


    template<int dim>
    void
    HeatEqn<dim>::make_grid() {
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
    HeatEqn<dim>::setup_level_set() {
        std::cout << "Setting up level set" << std::endl;

        // The level set function lives on the whole background mesh.
        levelset_dof_handler.distribute_dofs(fe_levelset);
        printf("leveset dofs: %d\n", levelset_dof_handler.n_dofs());
        levelset.reinit(levelset_dof_handler.n_dofs());

        // Project the geometry onto the mesh.
        VectorTools::project(levelset_dof_handler,
                             constraints,
                             QGauss<dim>(2 * element_order + 1),
                             *domain_function,
                             levelset);
    }

    template<int dim>
    void
    HeatEqn<dim>::distribute_dofs() {
        std::cout << "Distributing dofs" << std::endl;

        // We want to types of elements on the mesh
        // Lagrange elements and elements that are constant zero.
        fe_collection.push_back(fe);
        fe_collection.push_back(FE_Nothing<dim>());

        // TODO fiks dette for å få et sirkulært domene istedet.
        // Set outside finite elements to fe, and inside to FE_nothing
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            if (LocationToLevelSet::OUTSIDE ==
                cut_mesh_classifier.location_to_level_set(cell)) {
                // 1 is FE_nothing
                cell->set_active_fe_index(1);
            } else {
                // 0 is fe
                cell->set_active_fe_index(0);
            }
        }
        dof_handler.distribute_dofs(fe_collection);
    }

    template<int dim>
    void
    HeatEqn<dim>::initialize_matrices() {
        std::cout << "Initialize marices" << std::endl;
        solution.reinit(dof_handler.n_dofs());
        rhs.reinit(dof_handler.n_dofs());

        cutfem::nla::make_sparsity_pattern_for_stabilized(dof_handler,
                                                          sparsity_pattern);
        stiffness_matrix.reinit(sparsity_pattern);
    }

    template<int dim>
    void
    HeatEqn<dim>::assemble_matrix() {
        std::cout << "Assembling" << std::endl;

        stiffness_matrix = 0;
        rhs = 0;

        // Use a helper object to compute the stabilisation for both the velocity
        // and the pressure component.
        const FEValuesExtractors::Scalar velocities(0);
        stabilization::JumpStabilization<dim, FEValuesExtractors::Scalar>
                velocity_stabilization(dof_handler,
                                       mapping_collection,
                                       cut_mesh_classifier,
                                       constraints);
        if (stabilized) {
            velocity_stabilization.set_function_describing_faces_to_stabilize(
                    stabilization::inside_stabilization);
            velocity_stabilization.set_weight_function(
                    stabilization::taylor_weights);
            velocity_stabilization.set_extractor(velocities);
        }

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
        QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);
        FEFaceValues<dim> fe_face_values(fe,
                                         face_quadrature_formula,
                                         update_values | update_gradients |
                                         update_quadrature_points |
                                         update_normal_vectors |
                                         update_JxW_values);

        double beta_0 = 0.1;
        double gamma_A = beta_0 * element_order * (element_order + 1);
        double gamma_M = beta_0 * element_order * (element_order + 1);

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
                assemble_matrix_local_over_cell(*fe_values_bulk, loc2glb);
            }

            // Retrieve an FEValues object with quadrature points
            // on the immersed surface.
            const boost::optional<const FEImmersedSurfaceValues<dim> &>
                    fe_values_surface = cut_fe_values.get_surface_fe_values();

            if (fe_values_surface)
                assemble_matrix_local_over_surface(*fe_values_surface, loc2glb);

            if (stabilized) {
                // Compute and add the velocity stabilization.
                velocity_stabilization.compute_stabilization(cell);
                velocity_stabilization.add_stabilization_to_matrix(
                        gamma_M + tau * nu * gamma_A / (h * h),
                        stiffness_matrix);
            }
        }
    }

    template<int dim>
    void
    HeatEqn<dim>::assemble_matrix_local_over_cell(
            const FEValues<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {
        // TODO generelt: er det for mange hjelpeobjekter som opprettes her i cella?
        //  bør det heller gjøres i funksjonen før og sendes som argumenter? hvis
        //  det er mulig mtp cellene som blir cut da

        // Matrix and vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // Vector for values of the RightHandSide for all quadrature points on a cell.
        std::vector<double> rhs_values(fe_values.n_quadrature_points);
        rhs_function->value_list(fe_values.get_quadrature_points(), rhs_values);

        // Calculate often used terms in the beginning of each cell-loop
        std::vector<double> phi(dofs_per_cell);
        std::vector<Tensor<1, dim>> grad_phi(dofs_per_cell);

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
            for (const unsigned int k : fe_values.dof_indices()) {
                grad_phi[k] = fe_values.shape_grad(k, q);
                phi[k] = fe_values.shape_value(k, q);
            }

            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    local_matrix(i, j) += (bdf_coeffs[solutions.size()]
                                           * phi[j] * phi[i]
                                           +
                                           tau * nu * grad_phi[j] * grad_phi[i]
                                          ) * fe_values.JxW(q); // dx
                }
            }
        }
        stiffness_matrix.add(loc2glb, local_matrix);
    }


    template<int dim>
    void
    HeatEqn<dim>::assemble_matrix_local_over_surface(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {
        // Matrix and vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // Evaluate the boundary function for all quadrature points on this face.
        std::vector<double> bdd_values(fe_values.n_quadrature_points);
        boundary_values->value_list(fe_values.get_quadrature_points(),
                                    bdd_values);

        // Calculate often used terms in the beginning of each cell-loop
        std::vector<Tensor<1, dim>> grad_phi(dofs_per_cell);
        std::vector<double> phi(dofs_per_cell);

        double gamma = 20 * element_order * (element_order + 1);
        double mu = gamma / h;
        Tensor<1, dim> normal;

        for (unsigned int q : fe_values.quadrature_point_indices()) {
            normal = fe_values.normal_vector(q);

            for (const unsigned int k : fe_values.dof_indices()) {
                phi[k] = fe_values.shape_value(k, q);
                grad_phi[k] = fe_values.shape_grad(k, q);
            }

            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices()) {
                    local_matrix(i, j) +=
                            tau * nu * (mu * phi[j] * phi[i]  // mu (u, v)
                                        -
                                        grad_phi[j] * normal *
                                        phi[i] // (∂_n u,v)
                                        -
                                        phi[j] * grad_phi[i] *
                                        normal // (u,∂_n v)
                            ) * fe_values.JxW(q); // ds
                }
            }
        }
        stiffness_matrix.add(loc2glb, local_matrix);
    }

    template<int dim>
    void
    HeatEqn<dim>::assemble_rhs() {
        std::cout << "Assembling RHS" << std::endl;

        rhs = 0;

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
        QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);
        FEFaceValues<dim> fe_face_values(fe,
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

            if (fe_values_bulk) {
                assemble_rhs_local_over_cell(*fe_values_bulk, loc2glb);
            }

            // Retrieve an FEValues object with quadrature points
            // on the immersed surface.
            const boost::optional<const FEImmersedSurfaceValues<dim> &>
                    fe_values_surface = cut_fe_values.get_surface_fe_values();

            if (fe_values_surface)
                assemble_rhs_local_over_surface(*fe_values_surface, loc2glb);
        }
    }

    template<int dim>
    void
    HeatEqn<dim>::assemble_rhs_local_over_cell(
            const FEValues<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {
        // TODO generelt: er det for mange hjelpeobjekter som opprettes her i cella?
        //  bør det heller gjøres i funksjonen før og sendes som argumenter? hvis
        //  det er mulig mtp cellene som blir cut da

        // Matrix and vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
        // FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // Vector for values of the RightHandSide for all quadrature points on a cell.
        std::vector<double> rhs_values(fe_values.n_quadrature_points);
        rhs_function->value_list(fe_values.get_quadrature_points(), rhs_values);

        // Calculate often used terms in the beginning of each cell-loop
        std::vector<double> phi(dofs_per_cell);
        // std::vector<Tensor<1, dim>> grad_phi(dofs_per_cell);

        // Create vector of the previous solutions values
        // std::vector<double> val(fe_values.n_quadrature_points);
        std::vector<double> val(fe_values.n_quadrature_points, 0);
        std::vector<std::vector<double>> prev_solution_values(solutions.size(),
                                                              val);

        // The the values of the previous solutions, and insert into the
        // matrix initialized above.
        for (unsigned long i = 0; i < solutions.size(); ++i) {
            fe_values.get_function_values(solutions[i],
                                          prev_solution_values[i]);
        }

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
            for (const unsigned int i : fe_values.dof_indices()) {
                phi[i] = fe_values.shape_value(i, q);

                double prev_values = 0;
                for (unsigned long k = 0; k < solutions.size(); ++k) {
                    prev_values += bdf_coeffs[k] * prev_solution_values[k][q];
                }

                local_rhs(i) += (tau * rhs_values[q] * phi[i] // (f, v)
                                 - prev_values * phi[i]
                                ) * fe_values.JxW(q);      // dx
                // TODO add interpolation of the previous steps here, based on the length of solutions vector.
            }
        }
        rhs.add(loc2glb, local_rhs);
    }


    template<int dim>
    void
    HeatEqn<dim>::assemble_rhs_local_over_surface(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {
        // Matrix and vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        // Evaluate the boundary function for all quadrature points on this face.
        std::vector<double> bdd_values(fe_values.n_quadrature_points);
        boundary_values->value_list(fe_values.get_quadrature_points(),
                                    bdd_values);

        // Calculate often used terms in the beginning of each cell-loop
        std::vector<Tensor<1, dim>> grad_phi(dofs_per_cell);
        std::vector<double> phi(dofs_per_cell);

        double gamma = 20 * element_order * (element_order + 1);
        double mu = gamma / h;
        Tensor<1, dim> normal;

        for (unsigned int q : fe_values.quadrature_point_indices()) {
            normal = fe_values.normal_vector(q);
            for (const unsigned int i : fe_values.dof_indices()) {
                phi[i] = fe_values.shape_value(i, q);
                grad_phi[i] = fe_values.shape_grad(i, q);

                local_rhs(i) +=
                        tau * nu * (mu * bdd_values[q] * phi[i] // mu (g, v)
                                    -
                                    bdd_values[q] * grad_phi[i] *
                                    normal // (g, n ∂_n v)
                        ) * fe_values.JxW(q);        // ds
            }
        }
        rhs.add(loc2glb, local_rhs);
    }


    template<int dim>
    void
    HeatEqn<dim>::solve() {
        std::cout << "Solving system" << std::endl;
        SparseDirectUMFPACK inverse;
        inverse.initialize(stiffness_matrix);
        inverse.vmult(solution, rhs);
    }

    template<int dim>
    void HeatEqn<dim>::
    compute_condition_number() {
        std::cout << "Compute condition number" << std::endl;

        // Invert the stiffness_matrix
        FullMatrix<double> stiffness_matrix_full(solution.size());
        stiffness_matrix_full.copy_from(stiffness_matrix);
        FullMatrix<double> inverse(solution.size());
        inverse.invert(stiffness_matrix_full);

        double norm = stiffness_matrix.frobenius_norm();
        double inverse_norm = inverse.frobenius_norm();

        condition_number = norm * inverse_norm;
        std::cout << "  cond_num = " << condition_number << std::endl;

        // TODO bruk eigenvalues istedet
    }


    template<int dim>
    void
    HeatEqn<dim>::output_results(std::string &suffix) const {
        std::cout << "Output results" << std::endl;
        // Output results, see step-22
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "solution");
        data_out.build_patches();
        std::ofstream out("solution-d" + std::to_string(dim)
                          + "o" + std::to_string(element_order)
                          + "r" + std::to_string(n_refines) + suffix + ".vtk");
        data_out.write_vtk(out);

        // Output levelset function.
        DataOut<dim, DoFHandler<dim>> data_out_levelset;
        data_out_levelset.attach_dof_handler(levelset_dof_handler);
        data_out_levelset.add_data_vector(levelset, "levelset");
        data_out_levelset.build_patches();
        std::ofstream output_ls("levelset-d" + std::to_string(dim)
                                + "o" + std::to_string(element_order)
                                + "r" + std::to_string(n_refines) + suffix +
                                ".vtk");
        data_out_levelset.write_vtk(output_ls);
    }


    template<int dim>
    Error HeatEqn<dim>::compute_error() {

        double l2_error_integral;
        double h1_semi_error_integral;

        NonMatching::RegionUpdateFlags region_update_flags;
        region_update_flags.inside = update_values | update_JxW_values |
                                     update_gradients |
                                     update_quadrature_points;

        NonMatching::FEValues<dim> cut_fe_values(mapping_collection,
                                                 fe_collection,
                                                 q_collection,
                                                 q_collection1D,
                                                 region_update_flags,
                                                 cut_mesh_classifier,
                                                 levelset_dof_handler,
                                                 levelset);


        for (const auto &cell : dof_handler.active_cell_iterators()) {
            cut_fe_values.reinit(cell);

            // Retrieve an FEValues object with quadrature points
            // over the full cell.
            const boost::optional<const FEValues<dim> &> fe_values_bulk =
                    cut_fe_values.get_inside_fe_values();

            if (fe_values_bulk) {
                integrate_cell(*fe_values_bulk, l2_error_integral,
                               h1_semi_error_integral);
            }
        }

        Error error;
        error.mesh_size = h;
        error.l2_error = pow(l2_error_integral, 0.5);
        error.h1_semi = pow(h1_semi_error_integral, 0.5);
        error.h1_error = pow(l2_error_integral + h1_semi_error_integral, 0.5);
        error.cond_num = condition_number;
        return error;
    }


    template<int dim>
    Error HeatEqn<dim>::
    compute_time_error(std::vector<Error> errors) {
        double l2_error_integral = 0;
        double h1_error_integral = 0;

        double l_inf_l2 = 0;
        double l_inf_h1 = 0;

        for (Error error : errors) {
            l2_error_integral += tau * pow(error.l2_error, 2);
            h1_error_integral += tau * pow(error.h1_semi, 2);

            if (error.l2_error > l_inf_l2)
                l_inf_l2 = error.l2_error;
            if (error.h1_error > l_inf_h1)
                l_inf_h1 = error.h1_error;
        }

        Error error;
        error.mesh_size = h;
        error.time_step = tau;

        error.l2_error = pow(l2_error_integral, 0.5);
        error.h1_error = pow(l2_error_integral + h1_error_integral, 0.5);
        error.h1_semi = pow(h1_error_integral, 0.5);

        error.l_inf_l2_error = l_inf_l2;
        error.l_inf_h1_error = l_inf_h1;
        return error;
    }


    template<int dim>
    void HeatEqn<dim>::
    integrate_cell(const FEValues<dim> &fe_v,
                   double &l2_error_integral,
                   double &h1_error_integral) const {

        std::vector<double> solution_values(fe_v.n_quadrature_points);
        std::vector<Tensor<1, dim>> solution_gradients(
                fe_v.n_quadrature_points);
        std::vector<double> analytical_values(fe_v.n_quadrature_points);
        std::vector<Tensor<1, dim>> analytical_gradients(
                fe_v.n_quadrature_points);

        fe_v.get_function_values(solution, solution_values);
        fe_v.get_function_gradients(solution, solution_gradients);


        analytical_solution->value_list(fe_v.get_quadrature_points(),
                                        analytical_values);
        analytical_solution->gradient_list(fe_v.get_quadrature_points(),
                                           analytical_gradients);

        double diff_values;
        Tensor<1, dim> diff_gradients;
        for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q) {
            diff_values = analytical_values[q] - solution_values[q];
            diff_gradients = analytical_gradients[q] - solution_gradients[q];

            l2_error_integral += diff_values * diff_values * fe_v.JxW(q);
            h1_error_integral += diff_gradients * diff_gradients * fe_v.JxW(q);
        }

    }


    template<int dim>
    void HeatEqn<dim>::
    write_header_to_file(std::ofstream &file) {
        file << "h, \\|u\\|_{L^2}, \\|u\\|_{H^1}, |u|_{H^1}, \\kappa(A)"
             << std::endl;
    }


    template<int dim>
    void HeatEqn<dim>::
    write_error_to_file(Error &error, std::ofstream &file) {
        file << error.mesh_size << ","
             << error.l2_error << ","
             << error.h1_error << ","
             << error.h1_semi << ","
             << error.cond_num << std::endl;
    }


    template
    class HeatEqn<2>;

    template
    class HeatEqn<3>;


} // namespace examples::cut::HeatEquation