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
#include "flow_problem.h"


using namespace cutfem;

namespace utils::problems::flow {

    template<int dim>
    FlowProblem<dim>::
    FlowProblem(const unsigned int n_refines,
                const int element_order,
                const bool write_output,
                Function<dim> &levelset_func,
                TensorFunction<1, dim> &analytic_v,
                Function<dim> &analytic_p)
            : n_refines(n_refines),
              write_output(write_output),
              element_order(element_order),
              mixed_fe(FESystem<dim>(FE_Q<dim>(element_order + 1), dim),
                       1,
                       FE_Q<dim>(element_order),
                       1), fe_levelset(element_order),
              levelset_dof_handler(triangulation), dof_handler(triangulation),
              cut_mesh_classifier(triangulation, levelset_dof_handler,
                                  levelset) {
        h = 0;
        // Use no constraints when projecting.
        constraints.close();

        levelset_function = &levelset_func;
        analytical_velocity = &analytic_v;
        analytical_pressure = &analytic_p;
    }

    template<int dim>
    void FlowProblem<dim>::
    setup_quadrature() {
        const unsigned int quadOrder = 2 * element_order + 1;
        q_collection.push_back(QGauss<dim>(quadOrder));
        q_collection1D.push_back(QGauss<1>(quadOrder));
    }

    template<int dim>
    Error FlowProblem<dim>::
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
    void FlowProblem<dim>::
    setup_level_set() {
        std::cout << "Setting up level set" << std::endl;

        // The level set function lives on the whole background mesh.
        levelset_dof_handler.distribute_dofs(fe_levelset);
        printf("  leveset dofs: %d\n", levelset_dof_handler.n_dofs());
        levelset.reinit(levelset_dof_handler.n_dofs());

        // Project the geometry onto the mesh.
        VectorTools::project(levelset_dof_handler,
                             constraints,
                             QGauss<dim>(2 * element_order + 1),
                             *levelset_function,
                             levelset);
    }

    template<int dim>
    void FlowProblem<dim>::
    distribute_dofs() {
        std::cout << "Distributing dofs" << std::endl;

        // We want to types of elements on the mesh
        // Lagrange elements and elements that are constant zero..
        fe_collection.push_back(mixed_fe);
        fe_collection.push_back(FESystem<dim>(
                FESystem<dim>(FE_Nothing<dim>(), dim), 1, FE_Nothing<dim>(),
                1));

        // Set outside finite elements to mixed_fe, and inside to FE_nothing
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            if (LocationToLevelSet::OUTSIDE ==
                cut_mesh_classifier.location_to_level_set(cell)) {
                // 1 is FE_nothing
                cell->set_active_fe_index(1);
            } else {
                // 0 is mixed_fe
                cell->set_active_fe_index(0);
            }
        }
        dof_handler.distribute_dofs(fe_collection);
    }

    template<int dim>
    void
    FlowProblem<dim>::initialize_matrices() {
        std::cout << "Initialize marices" << std::endl;
        solution.reinit(dof_handler.n_dofs());
        rhs.reinit(dof_handler.n_dofs());

        cutfem::nla::make_sparsity_pattern_for_stabilized(dof_handler,
                                                          sparsity_pattern);
        stiffness_matrix.reinit(sparsity_pattern);
    }

    template<int dim>
    void FlowProblem<dim>::
    solve() {
        std::cout << "Solving system" << std::endl;
        SparseDirectUMFPACK inverse;
        inverse.initialize(stiffness_matrix);
        inverse.vmult(solution, rhs);
    }

    template<int dim>
    void FlowProblem<dim>::
    output_results(bool minimal_output) const {
        std::cout << "Output results" << std::endl;
        // Output results, see step-22
        std::vector<std::string> solution_names(dim, "velocity");
        solution_names.emplace_back("pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation> dci(
                dim, DataComponentInterpretation::component_is_part_of_vector);
        dci.push_back(DataComponentInterpretation::component_is_scalar);

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution,
                                 solution_names,
                                 DataOut<dim>::type_dof_data,
                                 dci);

        data_out.build_patches();
        std::ofstream output("solution-d" + std::to_string(dim)
                             + "o" + std::to_string(element_order)
                             + "r" + std::to_string(n_refines) + ".vtk");
        data_out.write_vtk(output);

        if (!minimal_output) {
            // Output levelset function.
            DataOut<dim, DoFHandler<dim>> data_out_levelset;
            data_out_levelset.attach_dof_handler(levelset_dof_handler);
            data_out_levelset.add_data_vector(levelset, "levelset");
            data_out_levelset.build_patches();
            std::ofstream output_ls("levelset-d" + std::to_string(dim)
                                    + "o" + std::to_string(element_order)
                                    + "r" + std::to_string(n_refines) + ".vtk");
            data_out_levelset.write_vtk(output_ls);
        }
    }


    template<int dim>
    Error FlowProblem<dim>::
    compute_error() {
        std::cout << "Compute error" << std::endl;
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
        error.h = h;
        error.l2_error_u = pow(l2_error_integral_u, 0.5);
        error.h1_error_u = pow(l2_error_integral_u + h1_error_integral_u, 0.5);
        error.h1_semi_u = pow(h1_error_integral_u, 0.5);
        error.l2_error_p = pow(l2_error_integral_p, 0.5);
        error.h1_error_p = pow(l2_error_integral_p + h1_error_integral_p, 0.5);
        error.h1_semi_p = pow(h1_error_integral_p, 0.5);
        return error;
    }


    template<int dim>
    void FlowProblem<dim>::
    integrate_cell(const FEValues<dim> &fe_v,
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


    template<int dim>
    void FlowProblem<dim>::
    write_header_to_file(std::ofstream &file) {
        file << "h, \\|u\\|_{L^2}, \\|u\\|_{H^1}, |u|_{H^1}, "
                "\\|p\\|_{L^2}, \\|p\\|_{H^1}, |p|_{H^1}" << std::endl;
    }


    template<int dim>
    void FlowProblem<dim>::
    write_error_to_file(Error &error, std::ofstream &file) {
        file << error.h << ","
             << error.l2_error_u << ","
             << error.h1_error_u << ","
             << error.h1_semi_u << ","
             << error.l2_error_p << ","
             << error.h1_error_p << ","
             << error.h1_semi_p << std::endl;
    }


    template
    class FlowProblem<2>;

    template
    class FlowProblem<3>;

} // namespace utils::problems::flow
