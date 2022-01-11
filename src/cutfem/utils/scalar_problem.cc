#include <deal.II/base/data_out_base.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/fe_immersed_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>

#include <assert.h>
#include <cmath>
#include <fstream>

#include "scalar_problem.h"


using namespace cutfem;
using namespace dealii;


namespace utils::problems::scalar {

    using NonMatching::FEImmersedSurfaceValues;


    template<int dim>
    ScalarProblem<dim>::ScalarProblem(const unsigned int n_refines,
                                      const int element_order,
                                      const bool write_output,
                                      LevelSet<dim> &levelset_func,
                                      Function<dim> &analytical_soln,
                                      const bool stabilized,
                                      const bool stationary,
                                      const bool compute_error)
            : CutFEMProblem<dim>(n_refines, element_order, write_output,
                                 levelset_func, stabilized, stationary,
                                 compute_error), fe(element_order) {
        analytical_solution = &analytical_soln;
    }


    template<int dim>
    void ScalarProblem<dim>::
    interpolate_solution(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                         int time_step) {
        // For time_step = 0, interpolate the boundary_values function, since
        // for t=0, this function should contain the initial values.
        if (time_step == 0) {
            VectorTools::interpolate(*dof_handler,
                                     *(this->boundary_values),
                                     this->solutions.front());
        } else {
            VectorTools::interpolate(*dof_handler,
                                     *(this->analytical_solution),
                                     this->solutions.front());
        }
    }


    template<int dim>
    void ScalarProblem<dim>::
    setup_fe_collection() {
        // We want to types of elements on the mesh
        // Lagrange elements and elements that are constant zero.
        if (this->fe_collection.size() == 0) {
            this->fe_collection.push_back(fe);
            this->fe_collection.push_back(FE_Nothing<dim>());
        }
    }


    template<int dim>
    void ScalarProblem<dim>::
    assemble_system() {
        std::cout << "Assembling scalar" << std::endl;

        this->stiffness_matrix = 0;
        this->rhs = 0;

        // Use a helper object to compute the stabilisation for both the velocity
        // and the pressure component.
        const FEValuesExtractors::Scalar velocities(0);
        stabilization::JumpStabilization<dim, FEValuesExtractors::Scalar>
                velocity_stabilization(*(this->dof_handlers.front()),
                                       this->mapping_collection,
                                       this->cut_mesh_classifier,
                                       this->constraints);
        if (this->stabilized) {
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

        NonMatching::FEValues<dim> cut_fe_values(this->mapping_collection,
                                                 this->fe_collection,
                                                 this->q_collection,
                                                 this->q_collection1D,
                                                 region_update_flags,
                                                 this->cut_mesh_classifier,
                                                 this->levelset_dof_handler,
                                                 this->levelset);

        // Quadrature for the faces of the cells on the outer boundary
        QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);
        FEFaceValues<dim> fe_face_values(fe,
                                         face_quadrature_formula,
                                         update_values | update_gradients |
                                         update_quadrature_points |
                                         update_normal_vectors |
                                         update_JxW_values);

        double beta_0 = 0.1;
        double gamma_A =
                beta_0 * this->element_order * (this->element_order + 1);
        double gamma_M =
                beta_0 * this->element_order * (this->element_order + 1);

        for (const auto &cell : this->dof_handlers.front()->active_cell_iterators()) {
            if (cell->is_locally_owned()) {
                const unsigned int n_dofs = cell->get_fe().dofs_per_cell;
                std::vector<types::global_dof_index> loc2glb(n_dofs);
                cell->get_dof_indices(loc2glb);

                // This call will compute quadrature rules relevant for this cell
                // in the background.
                cut_fe_values.reinit(cell);

                // Retrieve an FEValues object with quadrature points
                // over the full cell.
                const std_cxx17::optional<FEValues<dim>>& fe_values_bulk =
                        cut_fe_values.get_inside_fe_values();

                if (fe_values_bulk) {
                    this->assemble_local_over_cell(*fe_values_bulk, loc2glb);
                }

                // Retrieve an FEValues object with quadrature points
                // on the immersed surface.
                const std_cxx17::optional<FEImmersedSurfaceValues<dim>>&
                        fe_values_surface = cut_fe_values.get_surface_fe_values();

                if (fe_values_surface) {
                    this->assemble_local_over_surface(*fe_values_surface, loc2glb);
                }

                if (this->stabilized) {
                    // Compute and add the velocity stabilization.
                    velocity_stabilization.compute_stabilization(cell);
                    velocity_stabilization.add_stabilization_to_matrix(
                            gamma_M + gamma_A / (this->h * this->h),
                            this->stiffness_matrix);
                }
            }
        }
    }


    template<int dim>
    void ScalarProblem<dim>::
    assemble_rhs_and_bdf_terms_local_over_cell(
            const FEValues<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {
        // Vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
        Vector<double> local_rhs(dofs_per_cell);

        // Vector for values of the RightHandSide for all quadrature points on a cell.
        std::vector<double> rhs_values(fe_values.n_quadrature_points);
        this->rhs_function->value_list(fe_values.get_quadrature_points(),
                                       rhs_values);

        // Create vector of the previous solutions values
        std::vector<double> val(fe_values.n_quadrature_points, 0);
        std::vector<std::vector<double>> prev_solution_values(
                this->solutions.size(), val);

        // The the values of the previous solutions, and insert into the
        // matrix initialized above.
        for (unsigned long k = 1; k < this->solutions.size(); ++k) {
            fe_values.get_function_values(this->solutions[k],
                                          prev_solution_values[k]);
        }
        double phi_iq;
        double prev_values;
        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
            prev_values = 0;
            for (unsigned long k = 1; k < this->solutions.size(); ++k) {
                prev_values +=
                        this->bdf_coeffs[k] * prev_solution_values[k][q];
            }

            for (const unsigned int i : fe_values.dof_indices()) {
                phi_iq = fe_values.shape_value(i, q);
                local_rhs(i) += (this->tau * rhs_values[q] * phi_iq // (f, v)
                                 - prev_values * phi_iq       // (u_n, v)
                                ) * fe_values.JxW(q);         // dx
            }
        }
        this->rhs.add(loc2glb, local_rhs);
    }


    template<int dim>
    void ScalarProblem<dim>::
    assemble_rhs_and_bdf_terms_local_over_cell_moving_domain(
            const FEValues<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) {

        // TODO needed?
        const hp::FECollection<dim> &fe_collection = this->dof_handlers.front()->get_fe_collection();
        const hp::QCollection<dim> q_collection(fe_values.get_quadrature());

        // Vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_values.get_fe().dofs_per_cell;
        Vector<double> local_rhs(dofs_per_cell);

        // Vector for values of the RightHandSide for all quadrature points on a cell.
        std::vector<double> rhs_values(fe_values.n_quadrature_points);
        this->rhs_function->value_list(fe_values.get_quadrature_points(),
                                       rhs_values);

        // Create vector of the previous solutions values
        std::vector<double> val(fe_values.n_quadrature_points, 0);
        std::vector<std::vector<double>> prev_solution_values(
                this->solutions.size(), val);

        // The the values of the previous solutions, and insert into the
        // matrix initialized above.
        const typename Triangulation<dim>::active_cell_iterator &cell =
                fe_values.get_cell();
        hp::FEValues<dim> hp_fe_values(this->mapping_collection,
                                       fe_collection,
                                       q_collection,
                                       update_values);

        // Read out the solution values from the previous time steps that we
        // need for the BDF-method.
        for (unsigned long k = 1; k < this->solutions.size(); ++k) {
            typename hp::DoFHandler<dim>::active_cell_iterator cell_prev(
                    &(this->triangulation), cell->level(), cell->index(),
                    this->dof_handlers[k].get());
            const FiniteElement<dim> &fe = cell_prev->get_fe();
            if (fe.n_dofs_per_cell() == 0) {
                // This means that in the previous solution step, this cell had
                // FE_Nothing elements. We can therefore not use that cell to
                // get the values we need for the BDF-formula. If this happens
                // then the active mesh in the previous step(s) need to be
                // extended, such that the cells outside the physical domain
                // can be stabilized. When the aftive mesh is sufficiently
                // big in all time steps, we should never enter this clause.
                // If this happens, the values of 0 vill be used.
                std::cout << "# NB: need larger cell buffer outside the "
                             "physical domain." << std::endl;
            } else {
                // Get the function values from the previous time steps.
                // TODO check that this is actually done.
                hp_fe_values.reinit(cell_prev);
                const FEValues<dim> &fe_values_prev = hp_fe_values.get_present_fe_values();
                fe_values_prev.get_function_values(this->solutions[k],
                                                   prev_solution_values[k]);
            }
        }

        double phi_iq;
        double prev_values;
        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
            for (const unsigned int i : fe_values.dof_indices()) {
                prev_values = 0;
                for (unsigned long k = 1; k < this->solutions.size(); ++k) {
                    prev_values +=
                            this->bdf_coeffs[k] * prev_solution_values[k][q];
                }
                phi_iq = fe_values.shape_value(i, q);
                local_rhs(i) += (this->tau * rhs_values[q] * phi_iq // (f, v)
                                 - prev_values * phi_iq       // (u_n, v)
                                ) * fe_values.JxW(q);         // dx
            }
        }
        this->rhs.add(loc2glb, local_rhs);
    }


    template<int dim>
    ErrorBase *ScalarProblem<dim>::
    compute_error(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                  Vector<double> &solution) {
        std::cout << "Compute error" << std::endl;

        double l2_error_integral;
        double h1_semi_error_integral;

        NonMatching::RegionUpdateFlags region_update_flags;
        region_update_flags.inside = update_values | update_JxW_values |
                                     update_gradients |
                                     update_quadrature_points;

        // Use a quadrature of higher degree when computing the error, than
        // when the stiffness matrix is assembled. This is to make sure we do
        // not use the same quadrature points when computing the error, since
        // these points can get a better approximation than the other point in
        // the cell.
        const unsigned int n_quad_points = this->element_order + 3;
        hp::QCollection<dim> q_collection;
        q_collection.push_back(QGauss<dim>(n_quad_points));
        hp::QCollection<1> q_collection1D;
        q_collection1D.push_back(QGauss<1>(n_quad_points));

        NonMatching::FEValues<dim> cut_fe_values(this->mapping_collection,
                                                 this->fe_collection,
                                                 q_collection,
                                                 q_collection1D,
                                                 region_update_flags,
                                                 this->cut_mesh_classifier,
                                                 this->levelset_dof_handler,
                                                 this->levelset);

        for (const auto &cell : dof_handler->active_cell_iterators()) {
            if (cell->is_locally_owned()) {
                // TODO these computation needs to be fixed for mpirun
                cut_fe_values.reinit(cell);

                // Retrieve an FEValues object with quadrature points
                // over the full cell.
                const std_cxx17::optional<FEValues<dim>>& fe_values_bulk =
                        cut_fe_values.get_inside_fe_values();
                // TODO hva med intersected celler?

                if (fe_values_bulk) {
                    integrate_cell(*fe_values_bulk, solution, l2_error_integral,
                                h1_semi_error_integral);
                }
            }
        }

        auto *error = new ErrorScalar();
        error->h = this->h;
        error->tau = this->tau;
        error->l2_error = pow(l2_error_integral, 0.5);
        error->h1_semi = pow(h1_semi_error_integral, 0.5);
        error->h1_error = pow(l2_error_integral + h1_semi_error_integral, 0.5);
        return error;
    }


    template<int dim>
    ErrorBase *ScalarProblem<dim>::
    compute_time_error(std::vector<ErrorBase *> &errors) {
        double l2_error_integral = 0;
        double h1_error_integral = 0;

        double l_inf_l2 = 0;
        double l_inf_h1 = 0;

        for (ErrorBase *error : errors) {
            auto *err = dynamic_cast<ErrorScalar *>(error);
            l2_error_integral += this->tau * pow(err->l2_error, 2);
            h1_error_integral += this->tau * pow(err->h1_semi, 2);

            if (err->l2_error > l_inf_l2)
                l_inf_l2 = err->l2_error;
            if (err->h1_error > l_inf_h1)
                l_inf_h1 = err->h1_error;
        }

        auto *error = new ErrorScalar();
        error->h = this->h;
        error->tau = this->tau;

        error->l2_error = pow(l2_error_integral, 0.5);
        error->h1_error = pow(l2_error_integral + h1_error_integral, 0.5);
        error->h1_semi = pow(h1_error_integral, 0.5);

        error->l_inf_l2_error = l_inf_l2;
        error->l_inf_h1_error = l_inf_h1;
        return error;
    }


    template<int dim>
    void ScalarProblem<dim>::
    integrate_cell(const FEValues<dim> &fe_v,
                   Vector<double> &solution,
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
    void ScalarProblem<dim>::
    write_header_to_file(std::ofstream &file) {
        file << "h, \\tau, \\|u\\|_{L^2}, \\|u\\|_{H^1}, |u|_{H^1}, "
                "\\|u\\|_{l^\\infty L^2}, \\|u\\|_{l^\\infty H^1}, \\kappa(A)"
             << std::endl;
    }


    template<int dim>
    void ScalarProblem<dim>::
    write_error_to_file(ErrorBase *error, std::ofstream &file) {
        auto *err = dynamic_cast<ErrorScalar *>(error);
        file << err->h << ","
             << err->tau << ","
             << err->l2_error << ","
             << err->h1_error << ","
             << err->h1_semi << ","
             << err->l_inf_l2_error << ","
             << err->l_inf_h1_error << ","
             << err->cond_num << std::endl;
    }


    template<int dim>
    void ScalarProblem<dim>::
    write_time_header_to_file(std::ofstream &file) {
        file << "k, \\tau, h, \\|u\\|_{L^2}, \\|u\\|_{H^1}, "
                "|u|_{H^1}, \\kappa(A)"
             << std::endl;
    }


    template<int dim>
    void ScalarProblem<dim>::
    write_time_error_to_file(ErrorBase *error, std::ofstream &file) {
        auto *err = dynamic_cast<ErrorScalar *>(error);
        file << err->time_step << ","
             << err->tau << ","
             << err->h << ","
             << err->l2_error << ","
             << err->h1_error << ","
             << err->h1_semi << ","
             << err->cond_num << std::endl;
    }


    template<int dim>
    void ScalarProblem<dim>::
    output_results(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                   Vector<double> &solution,
                   std::string &suffix,
                   bool minimal_output) const {
        std::cout << "Output results" << std::endl;
        // Output results, see step-22
        DataOut<dim> data_out;
        data_out.attach_dof_handler(*dof_handler);
        data_out.add_data_vector(solution, "solution");
        data_out.build_patches();
        std::ofstream out("solution-d" + std::to_string(dim)
                          + "o" + std::to_string(this->element_order)
                          + "r" + std::to_string(this->n_refines)
                          + "-" + suffix + ".vtk");
        data_out.write_vtk(out);

        // Output levelset function.
        if (!minimal_output) {
            // TODO sett inn i egen funksjon
            DataOut<dim> data_out_levelset;
            data_out_levelset.attach_dof_handler(this->levelset_dof_handler);
            data_out_levelset.add_data_vector(this->levelset, "levelset");
            data_out_levelset.build_patches();
            std::ofstream output_ls("levelset-d" + std::to_string(dim)
                                    + "o" + std::to_string(this->element_order)
                                    + "r" + std::to_string(this->n_refines)
                                    + "-" + suffix + ".vtk");
            data_out_levelset.write_vtk(output_ls);
        }
    }


    template
    class ScalarProblem<2>;

    template
    class ScalarProblem<3>;


} // namespace utils::problems::scalar