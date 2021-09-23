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


#include "cutfem_problem.h"


namespace utils::problems {

    template<int dim>
    CutFEMProblem<dim>::
    CutFEMProblem(const unsigned int n_refines,
                  const int element_order,
                  const bool write_output,
                  Function<dim> &levelset_func,
                  const bool stabilized)
            : n_refines(n_refines), element_order(element_order),
              write_output(write_output),
              fe_levelset(element_order),
              levelset_dof_handler(triangulation),
              dof_handler(triangulation),
              cut_mesh_classifier(triangulation, levelset_dof_handler,
                                  levelset),
              stabilized(stabilized) {

        levelset_function = &levelset_func;
    }


    template<int dim>
    ErrorBase CutFEMProblem<dim>::
    run_step() {
        make_grid(triangulation);
        setup_quadrature();
        setup_level_set();
        cut_mesh_classifier.reclassify();
        distribute_dofs();
        initialize_matrices();
        assemble_system();
        solve();
        if (write_output) {
            output_results();
        }
        return compute_error();
    }


    template<int dim>
    Vector<double> CutFEMProblem<dim>::
    get_solution() {
        return solution;
    }


    template<int dim>
    ErrorBase CutFEMProblem<dim>::
    run_time(unsigned int bdf_type, unsigned int steps,
             std::vector<Vector<double>> &supplied_solutions) {

        std::cout << "\nBDF-" << bdf_type << ", steps=" << steps << std::endl;

        make_grid(this->triangulation);
        setup_quadrature();
        setup_level_set();
        cut_mesh_classifier.reclassify();
        distribute_dofs();
        initialize_matrices();

        // Vector for the computed error for each time step.
        std::vector<ErrorBase> errors(steps + 1);

        interpolate_first_steps(bdf_type, errors);
        set_supplied_solutions(bdf_type, supplied_solutions, errors);
        set_bdf_coefficients(bdf_type);

        assemble_matrix();

        std::ofstream file("errors-time-d" + std::to_string(dim)
                           + "o" + std::to_string(this->element_order)
                           + "r" + std::to_string(this->n_refines) + ".csv");
        write_time_header_to_file(file);

        // Write the interpolation errors to file.
        // TODO note that this results in both the interpolation error and the
        //  fem error to be written when u1 is supplied to bdf-2.
        for (unsigned int k = 0; k < bdf_type; ++k) {
            write_time_error_to_file(errors[k], file);
            errors[k].output();
        }

        double time;
        for (unsigned int k = bdf_type; k <= steps; ++k) {
            time = k * this->tau;
            std::cout << "\nTime Step = " << k
                      << ", tau = " << this->tau
                      << ", time = " << time << std::endl;

            /*
            this->rhs_function->set_time(time);
            this->boundary_values->set_time(time);
            this->analytical_velocity->set_time(time);
            analytical_pressure->set_time(time);
             */
            set_function_times(time);

            // TODO nødvendig??
            this->solution.reinit(this->solution.size());
            this->rhs.reinit(this->solution.size());

            assemble_rhs(k);
            this->solve();
            errors[k] = this->compute_error();
            errors[k].time_step = k;
            write_time_error_to_file(errors[k], file);

            if (this->write_output) {
                this->output_results(k, false);
            }

            for (unsigned long i = 1; i < solutions.size(); ++i) {
                solutions[i - 1] = solutions[i];
            }
            solutions[solutions.size() - 1] = this->solution;
        }

        std::cout << std::endl;
        for (ErrorBase error : errors) {
            error.output();
        }

        return compute_time_error(errors);
    }


    template<int dim>
    ErrorBase CutFEMProblem<dim>::
    run_time(unsigned int bdf_type, unsigned int steps) {
        // Invoking this method will result in a pure BDF-k method, where all
        // the initial steps will be interpolated.
        std::vector<Vector<double>> empty;
        return run_time(bdf_type, steps, empty);
    }


    template<int dim>
    void CutFEMProblem<dim>::
    set_bdf_coefficients(unsigned int bdf_type) {
        bdf_coeffs = std::vector<double>(bdf_type + 1);

        if (bdf_type == 1) {
            // BDF-1 (implicit Euler).
            bdf_coeffs[0] = -1;
            bdf_coeffs[1] = 1;
        } else if (bdf_type == 2) {
            // BDF-2.
            bdf_coeffs[0] = 0.5;
            bdf_coeffs[1] = -2;
            bdf_coeffs[2] = 1.5;
        } else if (bdf_type == 3) {
            bdf_coeffs[0] = -1.0 / 3;
            bdf_coeffs[1] = 1.5;
            bdf_coeffs[2] = -3;
            bdf_coeffs[3] = 11.0 / 6;
        } else {
            throw std::invalid_argument(
                    "bdf_type has to be either 1 or 2, not " +
                    std::to_string(bdf_type) + ".");
        }
    }

    /**
     *
     * @tparam dim
     * @param errors
     * @param u1
     * @param bdf_type
     *
     * The initial value u0 is interpolated using the boundary_values object,
     * while possibly next steps are interpolated using analytical_velocity and
     * analytical_pressure. This is becuase this is only done when u1 is not
     * supplied, which must mean we have the analytical solutions.
     *
     */
    template<int dim>
    void CutFEMProblem<dim>::
    interpolate_first_steps(unsigned int bdf_type,
                            std::vector<ErrorBase> &errors) {
        solutions = std::vector<Vector<double>>(bdf_type);

        std::cout << "Interpolate first step(s)." << std::endl;

        for (unsigned int k = 0; k < bdf_type; ++k) {
            //analytical_velocity->set_time(k * tau);
            // analytical_pressure->set_time(k * tau);
            set_function_times(k * this->tau);

            /*
            Utils::AnalyticalSolutionWrapper<dim> wrapper(*analytical_velocity,
                                                          *analytical_pressure);
            VectorTools::interpolate(dof_handler, wrapper, solution);
             */
            interpolate_solution(k);

            errors[k] = this->compute_error();
            errors[k].time_step = k;
            std::string suffix = "-" + std::to_string(k) + "-inter";
            this->output_results(suffix);

            solutions[k] = this->solution;
        }

        // TODO burde kanskje heller interpolere boundary_values for
        //  initial verdier?
        // Important that the boundary_values function uses t=0, when
        // we interpolate the initial value from it.
        // boundary_values->set_time(0);

        // Use the boundary_values as initial values. Interpolate the
        // boundary_values function into the finite element space.
        // const unsigned int n_components_on_element = dim + 1;
        // FEValuesExtractors::Vector velocities(0);
        //VectorFunctionFromTensorFunction<dim> adapter(
        //        *boundary_values,
        //        velocities.first_vector_component,
        //        n_components_on_element);

    }

    template<int dim>
    void CutFEMProblem<dim>::
    set_supplied_solutions(unsigned int bdf_type,
                           std::vector<Vector<double>> &supplied_solutions,
                           std::vector<ErrorBase> &errors) {
        std::cout << "Set supplied solutions" << std::endl;

        std::vector<Vector<double>> full_vector(bdf_type, Vector<double>(1));

        unsigned int size_diff = full_vector.size() - supplied_solutions.size();
        assert(size_diff >= 0);
        for (unsigned int k = 0; k < supplied_solutions.size(); ++k) {
            full_vector[size_diff + k] = supplied_solutions[k];
        }

        // The the supplied solutions in the solutions vector, and compute
        // the errors.
        for (unsigned int k = 0; k < bdf_type; ++k) {
            if (full_vector[k].size() == this->solution.size()) {
                std::cout << " - Set supplied for k = " << k << std::endl;
                solutions[k] = full_vector[k];
                // analytical_velocity->set_time(k * tau);
                // analytical_pressure->set_time(k * tau);
                set_function_times(k * this->tau);

                this->solution = full_vector[k];
                errors[k] = this->compute_error();
                errors[k].time_step = k;
            }
        }
    }


    template<int dim>
    void CutFEMProblem<dim>::
    set_function_times(double time) {
        throw std::logic_error(
                "Override this method to run a time dependent problem.");
    }


    template<int dim>
    void CutFEMProblem<dim>::
    interpolate_solution(int time_step) {
        throw std::logic_error(
                "Override this method to run a time dependent problem.");
    }


    template<int dim>
    void CutFEMProblem<dim>::
    setup_quadrature() {
        const unsigned int quadOrder = 2 * element_order + 1;
        q_collection.push_back(QGauss<dim>(quadOrder));
        q_collection1D.push_back(QGauss<1>(quadOrder));
    }


    template<int dim>
    void CutFEMProblem<dim>::
    setup_level_set() {
        std::cout << "Setting up level set" << std::endl;

        // The level set function lives on the whole background mesh.
        levelset_dof_handler.distribute_dofs(fe_levelset);
        printf("leveset dofs: %d\n", levelset_dof_handler.n_dofs());
        levelset.reinit(levelset_dof_handler.n_dofs());

        // Project the geometry onto the mesh.
        VectorTools::project(levelset_dof_handler,
                             constraints,
                             QGauss<dim>(2 * element_order + 1),
                             *levelset_function,
                             levelset);
    }


    template<int dim>
    void CutFEMProblem<dim>::
    initialize_matrices() {
        std::cout << "Initialize marices" << std::endl;
        solution.reinit(dof_handler.n_dofs());
        rhs.reinit(dof_handler.n_dofs());

        cutfem::nla::make_sparsity_pattern_for_stabilized(dof_handler,
                                                          sparsity_pattern);
        stiffness_matrix.reinit(sparsity_pattern);
    }


    template<int dim>
    void CutFEMProblem<dim>::
    solve() {
        std::cout << "Solving system" << std::endl;
        SparseDirectUMFPACK inverse;
        inverse.initialize(stiffness_matrix);
        inverse.vmult(solution, rhs);
    }


    template<int dim>
    double CutFEMProblem<dim>::
    compute_condition_number() {
        std::cout << "Compute condition number" << std::endl;

        // Invert the stiffness_matrix
        FullMatrix<double> stiffness_matrix_full(this->solution.size());
        stiffness_matrix_full.copy_from(this->stiffness_matrix);
        FullMatrix<double> inverse(this->solution.size());
        inverse.invert(stiffness_matrix_full);

        double norm = this->stiffness_matrix.frobenius_norm();
        double inverse_norm = inverse.frobenius_norm();

        double condition_number = norm * inverse_norm;
        std::cout << "  cond_num = " << condition_number << std::endl;

        // TODO bruk eigenvalues istedet
        return condition_number;
    }


    template<int dim>
    void CutFEMProblem<dim>::
    output_results(bool minimal_output) const {
        std::string empty = "";
        output_results(empty, minimal_output);
    }


    template<int dim>
    void CutFEMProblem<dim>::
    output_results(int time_step, bool minimal_output) const {
        std::string k = std::to_string(time_step);
        output_results(k, minimal_output);
    }

    /**
     * Compute the L2 and H1 error based on the computed error from each time
     * step.
     *
     * Compute the square root of the sum of the squared errors from each time
     * steps, weighted by the time step length tau for each term in the sum.
     */
    /*
    template<int dim>
    ErrorBase TimeProblem<dim>::
    compute_time_error(std::vector<Error> errors) {
        double l2_error_integral_u = 0;
        double h1_error_integral_u = 0;
        double l2_error_integral_p = 0;
        double h1_error_integral_p = 0;

        double l_inf_l2_u = 0;
        double l_inf_h1_u = 0;

        for (Error error : errors) {
            l2_error_integral_u += tau * pow(error.l2_error_u, 2);
            h1_error_integral_u += tau * pow(error.h1_semi_u, 2);
            l2_error_integral_p += tau * pow(error.l2_error_p, 2);
            h1_error_integral_p += tau * pow(error.h1_semi_p, 2);

            if (error.l2_error_u > l_inf_l2_u)
                l_inf_l2_u = error.l2_error_u;
            if (error.h1_error_u > l_inf_h1_u)
                l_inf_h1_u = error.h1_error_u;
        }

        Error error;
        error.mesh_size = h;
        error.tau = tau;

        error.l2_error_u = pow(l2_error_integral_u, 0.5);
        error.h1_error_u = pow(l2_error_integral_u + h1_error_integral_u, 0.5);
        error.h1_semi_u = pow(h1_error_integral_u, 0.5);
        error.l2_error_p = pow(l2_error_integral_p, 0.5);
        error.h1_error_p = pow(l2_error_integral_p + h1_error_integral_p, 0.5);
        error.h1_semi_p = pow(h1_error_integral_p, 0.5);

        error.l_inf_l2_error_u = l_inf_l2_u;
        error.l_inf_h1_error_u = l_inf_h1_u;
        return error;
    }

     */

    template
    class CutFEMProblem<2>;

    template
    class CutFEMProblem<3>;

}

