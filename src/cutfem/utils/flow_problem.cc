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

#include <cmath>
#include <fstream>

#include "cutfem/geometry/SignedDistanceSphere.h"
#include "cutfem/nla/sparsity_pattern.h"

#include "../../utils/integration.h"
#include "../../utils/output.h"
#include "flow_problem.h"


using namespace cutfem;

namespace utils::problems::flow {

    template<int dim>
    FlowProblem<dim>::
    FlowProblem(const unsigned int n_refines,
                const int element_order,
                const bool write_output,
                LevelSet<dim> &levelset_func,
                TensorFunction<1, dim> &analytic_v,
                Function<dim> &analytic_p,
                const bool stabilized,
                const bool stationary,
                const bool compute_error)
            : CutFEMProblem<dim>(n_refines, element_order, write_output,
                                 levelset_func, stabilized, stationary,
                                 compute_error),
              mixed_fe(FESystem<dim>(FE_Q<dim>(element_order + 1), dim), 1,
                       FE_Q<dim>(element_order), 1) {
        analytical_velocity = &analytic_v;
        analytical_pressure = &analytic_p;
    }


    template<int dim>
    void FlowProblem<dim>::
    interpolate_solution(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                         int time_step) {
        if (time_step == 0) {
            // Use the boundary_values as initial values. Interpolate the
            // boundary_values function into the finite element space.
            boundary_values->set_time(0);
            Utils::AnalyticalSolutionWrapper<dim> wrapper(*boundary_values,
                                                          *analytical_pressure);
            VectorTools::interpolate(*dof_handler, wrapper,
                                     this->solutions.front());
        } else {
            Utils::AnalyticalSolutionWrapper<dim> wrapper(*analytical_velocity,
                                                          *analytical_pressure);
            VectorTools::interpolate(*dof_handler, wrapper,
                                     this->solutions.front());
        }
    }


    template<int dim>
    void FlowProblem<dim>::
    setup_fe_collection() {
        // We want to types of elements on the mesh
        // Lagrange elements and elements that are constant zero..
        this->fe_collection.push_back(mixed_fe);
        this->fe_collection.push_back(
                FESystem<dim>(FESystem<dim>(FE_Nothing<dim>(), dim), 1,
                              FE_Nothing<dim>(), 1));
    }


    template<int dim>
    void FlowProblem<dim>::
    setup_quadrature() {
        const unsigned int n_quad_points = this->element_order + 2;
        this->q_collection.push_back(QGauss<dim>(n_quad_points));
        this->q_collection1D.push_back(QGauss<1>(n_quad_points));
    }


    template<int dim>
    void FlowProblem<dim>::
    assemble_rhs_and_bdf_terms_local_over_cell(
            const FEValues<dim> &fe_v,
            const std::vector<types::global_dof_index> &loc2glb) {
        // Vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_v.get_fe().dofs_per_cell;
        Vector<double> local_rhs(dofs_per_cell);

        // Vector for values of the RightHandSide for all quadrature points on a cell.
        std::vector<Tensor<1, dim>> rhs_values(fe_v.n_quadrature_points,
                                               Tensor<1, dim>());
        this->rhs_function->value_list(fe_v.get_quadrature_points(),
                                       rhs_values);

        const FEValuesExtractors::Vector v(0);

        std::vector<Tensor<1, dim>> val(fe_v.n_quadrature_points,
                                        Tensor<1, dim>());
        std::vector<std::vector<Tensor<1, dim >>> prev_solutions_values(
                this->solutions.size(), val);

        for (unsigned int k = 0; k < this->solutions.size(); ++k) {
            fe_v[v].get_function_values(this->solutions[k],
                                        prev_solutions_values[k]);
        }

        Tensor<1, dim> phi_u;
        Tensor<1, dim> prev_values;

        const int time_switch = this->stationary ? 0 : 1;

        for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q) {
            // RHS
            prev_values = Tensor<1, dim>();
            for (unsigned int k = 1; k < this->solutions.size(); ++k) {
                prev_values +=
                        this->bdf_coeffs[k] * prev_solutions_values[k][q];
            }

            for (const unsigned int i : fe_v.dof_indices()) {

                phi_u = fe_v[v].value(i, q);
                local_rhs(i) += (this->tau * rhs_values[q] * phi_u    // τ(f, v)
                                 - prev_values * phi_u * time_switch
                                ) * fe_v.JxW(q);      // dx
            }
        }
        this->rhs.add(loc2glb, local_rhs);
    }


    template<int dim>
    void FlowProblem<dim>::
    assemble_rhs_and_bdf_terms_local_over_cell_moving_domain(
            const FEValues<dim> &fe_v,
            const std::vector<types::global_dof_index> &loc2glb) {

        // TODO needed?
        const hp::FECollection<dim> &fe_collection = this->dof_handlers.front()->get_fe_collection();
        const hp::QCollection<dim> q_collection(fe_v.get_quadrature());

        // Vector for the contribution of each cell
        const unsigned int dofs_per_cell = fe_v.get_fe().dofs_per_cell;
        Vector<double> local_rhs(dofs_per_cell);

        const FEValuesExtractors::Vector v(0);

        // Vector for values of the RightHandSide for all quadrature points on a cell.
        std::vector<Tensor<1, dim>> rhs_values(fe_v.n_quadrature_points,
                                               Tensor<1, dim>());
        this->rhs_function->value_list(fe_v.get_quadrature_points(),
                                       rhs_values);

        // Create vector of the previous solutions values
        std::vector<Tensor<1, dim>> val(fe_v.n_quadrature_points,
                                        Tensor<1, dim>());
        std::vector<std::vector<Tensor<1, dim>>> prev_solution_values(
                this->solutions.size(), val);

        // The the values of the previous solutions, and insert into the
        // matrix initialized above.
        const typename Triangulation<dim>::active_cell_iterator &cell =
                fe_v.get_cell();
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
                hp_fe_values.reinit(cell_prev);
                const FEValues<dim> &fe_values_prev = hp_fe_values.get_present_fe_values();
                fe_values_prev[v].get_function_values(this->solutions[k],
                                                      prev_solution_values[k]);
            }

        }
        Tensor<1, dim> phi_u;
        Tensor<1, dim> prev_values;
        const int time_switch = this->stationary ? 0 : 1;

        for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q) {
            prev_values = 0;
            for (unsigned long k = 1; k < this->solutions.size(); ++k) {
                prev_values +=
                        this->bdf_coeffs[k] * prev_solution_values[k][q];
            }
            for (const unsigned int i : fe_v.dof_indices()) {
                phi_u = fe_v[v].value(i, q);
                local_rhs(i) += (this->tau * rhs_values[q] * phi_u   // (f, v)
                                 - prev_values * phi_u * time_switch // (u_n, v)
                                ) * fe_v.JxW(q);         // dx
            }
        }
        this->rhs.add(loc2glb, local_rhs);
    }


    template<int dim>
    ErrorBase *FlowProblem<dim>::
    compute_error(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                  Vector<double> &solution) {
        std::cout << "Compute error" << std::endl;
        NonMatching::RegionUpdateFlags region_update_flags;
        region_update_flags.inside = update_values | update_JxW_values |
                                     update_gradients |
                                     update_quadrature_points;
        region_update_flags.surface = update_values | update_JxW_values |
                                      update_gradients |
                                      update_quadrature_points |
                                      update_normal_vectors;

        // Use a higher order quadrature formula when computing the error than
        // when assembling the stiffness matrix.
        const unsigned int n_quad_points = this->element_order + 4;
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

        // Compute the mean of the numerical and the exact pressure over the
        // domain, to subtract it before computing the error.
        double mean_num_pressure = 0;
        double mean_ext_pressure = 0;
        Utils::compute_mean_pressure(*dof_handler,
                                     cut_fe_values,
                                     solution,
                                     *analytical_pressure,
                                     mean_num_pressure,
                                     mean_ext_pressure);

        double l2_error_integral_u = 0;
        double h1_error_integral_u = 0;
        double l2_error_integral_p = 0;
        double h1_error_integral_p = 0;

        for (const auto &cell : dof_handler->active_cell_iterators()) {
            cut_fe_values.reinit(cell);

            const boost::optional<const FEValues<dim> &> fe_values_inside =
                    cut_fe_values.get_inside_fe_values();

            if (fe_values_inside) {
                integrate_cell(*fe_values_inside, solution,
                               l2_error_integral_u,
                               h1_error_integral_u, l2_error_integral_p,
                               h1_error_integral_p, mean_num_pressure,
                               mean_ext_pressure);
            }
        }

        ErrorFlow *error = new ErrorFlow();
        error->h = this->h;
        error->tau = this->tau;
        error->l2_error_u = pow(l2_error_integral_u, 0.5);
        error->h1_error_u = pow(l2_error_integral_u + h1_error_integral_u,
                                0.5);
        error->h1_semi_u = pow(h1_error_integral_u, 0.5);
        error->l2_error_p = pow(l2_error_integral_p, 0.5);
        error->h1_error_p = pow(l2_error_integral_p + h1_error_integral_p,
                                0.5);
        error->h1_semi_p = pow(h1_error_integral_p, 0.5);
        return error;
    }


    /**
     * Compute the L2 and H1 error based on the computed error from each time
     * step.
     *
     * Compute the square root of the sum of the squared errors from each time
     * steps, weighted by the time step length tau for each term in the sum.
     */
    template<int dim>
    ErrorBase *FlowProblem<dim>::
    compute_time_error(std::vector<ErrorBase *> &errors) {
        double l2_error_integral_u = 0;
        double h1_error_integral_u = 0;
        double l2_error_integral_p = 0;
        double h1_error_integral_p = 0;

        double l_inf_l2_u = 0;
        double l_inf_h1_u = 0;

        for (ErrorBase *error : errors) {
            auto *err = dynamic_cast<ErrorFlow *>(error);
            l2_error_integral_u += this->tau * pow(err->l2_error_u, 2);
            h1_error_integral_u += this->tau * pow(err->h1_semi_u, 2);
            l2_error_integral_p += this->tau * pow(err->l2_error_p, 2);
            h1_error_integral_p += this->tau * pow(err->h1_semi_p, 2);

            if (err->l2_error_u > l_inf_l2_u)
                l_inf_l2_u = err->l2_error_u;
            if (err->h1_error_u > l_inf_h1_u)
                l_inf_h1_u = err->h1_error_u;
        }

        auto *error = new ErrorFlow();
        error->h = this->h;
        error->tau = this->tau;

        error->l2_error_u = pow(l2_error_integral_u, 0.5);
        error->h1_error_u = pow(l2_error_integral_u + h1_error_integral_u,
                                0.5);
        error->h1_semi_u = pow(h1_error_integral_u, 0.5);
        error->l2_error_p = pow(l2_error_integral_p, 0.5);
        error->h1_error_p = pow(l2_error_integral_p + h1_error_integral_p,
                                0.5);
        error->h1_semi_p = pow(h1_error_integral_p, 0.5);

        error->l_inf_l2_error_u = l_inf_l2_u;
        error->l_inf_h1_error_u = l_inf_h1_u;
        return error;
    }


    template<int dim>
    void FlowProblem<dim>::
    integrate_cell(const FEValues<dim> &fe_v,
                   Vector<double> &solution,
                   double &l2_error_integral_u,
                   double &h1_error_integral_u,
                   double &l2_error_integral_p,
                   double &h1_error_integral_p,
                   const double &mean_numerical_pressure,
                   const double &mean_exact_pressure) const {

        const FEValuesExtractors::Vector v(0);
        const FEValuesExtractors::Scalar p(dim);

        std::vector<Tensor<1, dim>> u_solution_values(
                fe_v.n_quadrature_points);
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
        std::vector<Tensor<2, dim>> u_exact_gradients(
                fe_v.n_quadrature_points,
                Tensor<2, dim>());
        std::vector<Tensor<1, dim>> p_exact_gradients(
                fe_v.n_quadrature_points,
                Tensor<1, dim>());
        analytical_velocity->gradient_list(fe_v.get_quadrature_points(),
                                           u_exact_gradients);
        analytical_pressure->gradient_list(fe_v.get_quadrature_points(),
                                           p_exact_gradients);

        for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q) {
            // Integrate the square difference between exact and numeric solution
            // for function values and gradients (both pressure and velocity).
            Tensor<1, dim> diff_u =
                    u_exact_solution[q] - u_solution_values[q];
            double diff_p = (p_exact_solution[q] - mean_exact_pressure) -
                            (p_solution_values[q] -
                             mean_numerical_pressure);

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


    template<int dim>
    void FlowProblem<dim>::
    write_time_header_to_file(std::ofstream &file) {
        file << "k, \\tau, h, \\|u\\|_{L^2}, \\|u\\|_{H^1}, |u|_{H^1}, "
                "\\|p\\|_{L^2}, \\|p\\|_{H^1}, |p|_{H^1}, \\kappa(A)"
             << std::endl;
    }


    template<int dim>
    void FlowProblem<dim>::
    write_time_error_to_file(ErrorBase *error, std::ofstream &file) {
        auto *err = dynamic_cast<ErrorFlow *>(error);
        file << err->time_step << ","
             << err->tau << ","
             << err->h << ","
             << err->l2_error_u << ","
             << err->h1_error_u << ","
             << err->h1_semi_u << ","
             << err->l2_error_p << ","
             << err->h1_error_p << ","
             << err->h1_semi_p << ","
             << err->cond_num << std::endl;
    }


    template<int dim>
    void FlowProblem<dim>::
    output_results(std::shared_ptr<hp::DoFHandler<dim>> &dof_handler,
                   Vector<double> &solution,
                   std::string &suffix,
                   bool minimal_output) const {
        std::cout << "Output results flow" << std::endl;
        // Output results, see step-22

        std::ofstream output("solution-d" + std::to_string(dim)
                             + "o" + std::to_string(this->element_order)
                             + "r" + std::to_string(this->n_refines)
                             + "-" + suffix + ".vtk");
        Utils::writeNumericalSolution(*dof_handler, solution, output);


        std::ofstream output_ex("analytical-d" + std::to_string(dim)
                                + "o" + std::to_string(this->element_order)
                                + "r" + std::to_string(this->n_refines)
                                + "-" + suffix + ".vtk");
        std::ofstream file_diff("diff-d" + std::to_string(dim)
                                + "o" + std::to_string(this->element_order)
                                + "r" + std::to_string(this->n_refines)
                                + "-" + suffix + ".vtk");
        Utils::writeAnalyticalSolutionAndDiff(*dof_handler,
                                              this->fe_collection,
                                              solution,
                                              *analytical_velocity,
                                              *analytical_pressure,
                                              output_ex,
                                              file_diff);

        if (!minimal_output) {
            // Output levelset function.
            DataOut<dim, DoFHandler<dim>> data_out_levelset;
            data_out_levelset.attach_dof_handler(
                    this->levelset_dof_handler);
            data_out_levelset.add_data_vector(this->levelset, "levelset");
            data_out_levelset.build_patches();
            std::ofstream output_ls("levelset-d" + std::to_string(dim)
                                    + "o" +
                                    std::to_string(this->element_order)
                                    + "r" + std::to_string(this->n_refines)
                                    + "-" + suffix + ".vtk");
            data_out_levelset.write_vtk(output_ls);
        }
    }


    template
    class FlowProblem<2>;

    template
    class FlowProblem<3>;

} // namespace utils::problems::flow
