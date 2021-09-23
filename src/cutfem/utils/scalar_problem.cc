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

#include <deal.II/non_matching/cut_mesh_classifier.h>
#include <deal.II/non_matching/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/optional.hpp>

#include <assert.h>
#include <cmath>
#include <fstream>

#include "scalar_problem.h"


using namespace cutfem;


namespace utils::problems::scalar {


    template<int dim>
    ScalarProblem<dim>::ScalarProblem(const unsigned int n_refines,
                                      const int element_order,
                                      const bool write_output,
                                      Function<dim> &levelset_func,
                                      Function<dim> &analytical_soln,
                                      const bool stabilized)
            : CutFEMProblem<dim>(n_refines, element_order,
                                 write_output, levelset_func, stabilized),
              fe(element_order) {
        // Use no constraints when projecting.
        this->constraints.close();

        analytical_solution = &analytical_soln;
    }


    template<int dim>
    void ScalarProblem<dim>::
    distribute_dofs() {
        std::cout << "Distributing dofs" << std::endl;

        // We want to types of elements on the mesh
        // Lagrange elements and elements that are constant zero.
        this->fe_collection.push_back(fe);
        this->fe_collection.push_back(FE_Nothing<dim>());

        // TODO fiks dette for å få et sirkulært domene istedet.
        // Set outside finite elements to fe, and inside to FE_nothing
        for (const auto &cell : this->dof_handler.active_cell_iterators()) {
            if (LocationToLevelSet::OUTSIDE ==
                this->cut_mesh_classifier.location_to_level_set(cell)) {
                // 1 is FE_nothing
                cell->set_active_fe_index(1);
            } else {
                // 0 is fe
                cell->set_active_fe_index(0);
            }
        }
        this->dof_handler.distribute_dofs(this->fe_collection);
    }


    template<int dim>
    void ScalarProblem<dim>::
    assemble_system() {
        std::cout << "Assembling" << std::endl;

        this->stiffness_matrix = 0;
        this->rhs = 0;

        // Use a helper object to compute the stabilisation for both the velocity
        // and the pressure component.
        const FEValuesExtractors::Scalar velocities(0);
        stabilization::JumpStabilization<dim, FEValuesExtractors::Scalar>
                velocity_stabilization(this->dof_handler,
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

        for (const auto &cell : this->dof_handler.active_cell_iterators()) {
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
                this->assemble_local_over_cell(*fe_values_bulk, loc2glb);
            }

            // Retrieve an FEValues object with quadrature points
            // on the immersed surface.
            const boost::optional<const FEImmersedSurfaceValues<dim> &>
                    fe_values_surface = cut_fe_values.get_surface_fe_values();

            if (fe_values_surface)
                this->assemble_local_over_surface(*fe_values_surface, loc2glb);

            if (this->stabilized) {
                // Compute and add the velocity stabilization.
                velocity_stabilization.compute_stabilization(cell);
                velocity_stabilization.add_stabilization_to_matrix(
                        gamma_M + gamma_A / (this->h * this->h),
                        this->stiffness_matrix);
            }
        }
    }


    template<int dim>
    void ScalarProblem<dim>::
    output_results(std::string &suffix,
                   bool minimal_output) const {
        std::cout << "Output results" << std::endl;
        // Output results, see step-22
        DataOut<dim> data_out;
        data_out.attach_dof_handler(this->dof_handler);
        data_out.add_data_vector(this->solution, "solution");
        data_out.build_patches();
        std::ofstream out("solution-d" + std::to_string(dim)
                          + "o" + std::to_string(this->element_order)
                          + "r" + std::to_string(this->n_refines) + suffix +
                          ".vtk");
        data_out.write_vtk(out);

        // Output levelset function.
        if (!minimal_output) {
            DataOut<dim, DoFHandler<dim>> data_out_levelset;
            data_out_levelset.attach_dof_handler(this->levelset_dof_handler);
            data_out_levelset.add_data_vector(this->levelset, "levelset");
            data_out_levelset.build_patches();
            std::ofstream output_ls("levelset-d" + std::to_string(dim)
                                    + "o" + std::to_string(this->element_order)
                                    + "r" + std::to_string(this->n_refines) +
                                    suffix +
                                    ".vtk");
            data_out_levelset.write_vtk(output_ls);
        }
    }


    template<int dim>
    ErrorBase ScalarProblem<dim>::
    compute_error() {
        // TODO bør jeg heller returnere en peker?

        double l2_error_integral;
        double h1_semi_error_integral;

        NonMatching::RegionUpdateFlags region_update_flags;
        region_update_flags.inside = update_values | update_JxW_values |
                                     update_gradients |
                                     update_quadrature_points;

        NonMatching::FEValues<dim> cut_fe_values(this->mapping_collection,
                                                 this->fe_collection,
                                                 this->q_collection,
                                                 this->q_collection1D,
                                                 region_update_flags,
                                                 this->cut_mesh_classifier,
                                                 this->levelset_dof_handler,
                                                 this->levelset);


        for (const auto &cell : this->dof_handler.active_cell_iterators()) {
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

        ErrorScalar error;
        error.h = this->h;
        error.tau = this->tau;
        error.l2_error = pow(l2_error_integral, 0.5);
        error.h1_semi = pow(h1_semi_error_integral, 0.5);
        error.h1_error = pow(l2_error_integral + h1_semi_error_integral, 0.5);
        return error;
    }


    template<int dim>
    ErrorBase ScalarProblem<dim>::
    compute_time_error(std::vector<ErrorBase> errors) {
        double l2_error_integral = 0;
        double h1_error_integral = 0;

        double l_inf_l2 = 0;
        double l_inf_h1 = 0;

        for (ErrorBase error : errors) {
            auto &err = dynamic_cast<ErrorScalar&>(error);
            l2_error_integral += this->tau * pow(err.l2_error, 2);
            h1_error_integral += this->tau * pow(err.h1_semi, 2);

            if (err.l2_error > l_inf_l2)
                l_inf_l2 = err.l2_error;
            if (err.h1_error > l_inf_h1)
                l_inf_h1 = err.h1_error;
        }

        ErrorScalar error;
        error.h = this->h;
        error.tau = this->tau;

        error.l2_error = pow(l2_error_integral, 0.5);
        error.h1_error = pow(l2_error_integral + h1_error_integral, 0.5);
        error.h1_semi = pow(h1_error_integral, 0.5);

        error.l_inf_l2_error = l_inf_l2;
        error.l_inf_h1_error = l_inf_h1;
        return error;
    }


    template<int dim>
    void ScalarProblem<dim>::
    integrate_cell(const FEValues<dim> &fe_v,
                   double &l2_error_integral,
                   double &h1_error_integral) const {

        std::vector<double> solution_values(fe_v.n_quadrature_points);
        std::vector<Tensor<1, dim>> solution_gradients(
                fe_v.n_quadrature_points);
        std::vector<double> analytical_values(fe_v.n_quadrature_points);
        std::vector<Tensor<1, dim>> analytical_gradients(
                fe_v.n_quadrature_points);

        fe_v.get_function_values(this->solution, solution_values);
        fe_v.get_function_gradients(this->solution, solution_gradients);


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
        file
                << "h, \\tau, \\|u\\|_{L^2}, \\|u\\|_{H^1}, |u|_{H^1}, \\|u\\|_{l^\\infty L^2}, \\|u\\|_{l^\\infty H^1}, \\kappa(A)"
                << std::endl;
    }


    template<int dim>
    void ScalarProblem<dim>::
    write_error_to_file(ErrorBase &error, std::ofstream &file) {
        auto &err = dynamic_cast<ErrorScalar&>(error);
        file << err.h << ","
             << err.tau << ","
             << err.l2_error << ","
             << err.h1_error << ","
             << err.h1_semi << ","
             << err.l_inf_l2_error << ","
             << err.l_inf_h1_error << ","
             << err.cond_num << std::endl;
    }


    template<int dim>
    void ScalarProblem<dim>::
    write_time_header_to_file(std::ofstream &file) {
        file << "k, \\|u\\|_{L^2}, \\|u\\|_{H^1}, |u|_{H^1}, \\kappa(A)"
             << std::endl;
    }


    template<int dim>
    void ScalarProblem<dim>::
    write_time_error_to_file(ErrorBase &error, std::ofstream &file) {
        auto &err = dynamic_cast<ErrorScalar&>(error);
        file << err.time_step << ","
             << err.l2_error << ","
             << err.h1_error << ","
             << err.h1_semi << ","
             << err.cond_num << std::endl;
    }


    template
    class ScalarProblem<2>;

    template
    class ScalarProblem<3>;


} // namespace utils::problems::scalar