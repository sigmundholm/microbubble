#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>

#include "ErrorStokesCylinder.h"


using namespace dealii;


template<int dim>
ErrorStokesCylinder<dim>::ErrorStokesCylinder(const double radius,
                                              const double half_length,
                                              const unsigned int n_refines,
                                              const int element_order,
                                              const bool write_output,
                                              StokesRhs<dim> &rhs_function,
                                              BoundaryValues<dim> &boundary_values,
                                              AnalyticalSolution<dim> &analytical_soln,
                                              const double pressure_drop,  // TODO remove
                                              const double sphere_radius,
                                              const double sphere_x_coord)
        : StokesCylinder<dim>(radius, half_length, n_refines, element_order,
                              write_output, rhs_function, boundary_values,
                              sphere_radius,
                              sphere_x_coord),
          pressure_drop(pressure_drop) {

    // Set to some unused boundary_id for Dirichlet only
    // TODO fiks riktige outflow betingelser
    this->do_nothing_id = 1000;
    analytical_solution = &analytical_soln;
}


template<int dim>
Error ErrorStokesCylinder<dim>::
compute_error() {
    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_JxW_values |
                                 update_gradients | update_quadrature_points;
    region_update_flags.surface = update_values | update_JxW_values |
                                  update_gradients | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> cut_fe_values(this->mapping_collection,
                                             this->fe_collection,
                                             this->q_collection,
                                             this->q_collection1D,
                                             region_update_flags,
                                             this->cut_mesh_classifier,
                                             this->levelset_dof_handler,
                                             this->levelset);

    double l2_error_integral_u = 0;
    double h1_error_integral_u = 0;
    double l2_error_integral_p = 0;
    double h1_error_integral_p = 0;

    for (const auto &cell : this->dof_handler.active_cell_iterators()) {
        cut_fe_values.reinit(cell);

        const boost::optional<const FEValues<dim> &> fe_values_inside =
                cut_fe_values.get_inside_fe_values();

        if (fe_values_inside) {
            integrate_cell(*fe_values_inside, l2_error_integral_u,
                           h1_error_integral_u, l2_error_integral_p,
                           h1_error_integral_p);
        }
    }

    Error error;
    error.mesh_size = this->h;
    error.l2_error_u = pow(l2_error_integral_u, 0.5);
    error.h1_error_u = pow(l2_error_integral_u + h1_error_integral_u, 0.5);
    error.h1_semi_u = pow(h1_error_integral_u, 0.5);
    error.l2_error_p = pow(l2_error_integral_p, 0.5);
    error.h1_error_p = pow(l2_error_integral_p + h1_error_integral_p, 0.5);
    error.h1_semi_p = pow(h1_error_integral_p, 0.5);
    return error;
}


template<int dim>
void ErrorStokesCylinder<dim>::
integrate_cell(const FEValues<dim> &fe_values,
               double &l2_error_integral_u,
               double &h1_error_integral_u,
               double &l2_error_integral_p,
               double &h1_error_integral_p) const {

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    std::vector<Tensor<1, dim>> u_solution_values(
            fe_values.n_quadrature_points);
    std::vector<Tensor<2, dim>> u_solution_gradients(
            fe_values.n_quadrature_points);
    std::vector<double> p_solution_values(fe_values.n_quadrature_points);
    std::vector<Tensor<1, dim>> p_solution_gradients(
            fe_values.n_quadrature_points);

    fe_values[velocities].get_function_values(this->solution,
                                              u_solution_values);
    fe_values[velocities].get_function_gradients(this->solution,
                                                 u_solution_gradients);
    fe_values[pressure].get_function_values(this->solution, p_solution_values);
    fe_values[pressure].get_function_gradients(this->solution,
                                               p_solution_gradients);

    // Exact solution: velocity and pressure
    std::vector<Tensor<1, dim>> u_exact_solution(fe_values.n_quadrature_points,
                                                 Tensor<1, dim>());
    std::vector<double> p_exact_solution(fe_values.n_quadrature_points);
    analytical_solution->value_list(fe_values.get_quadrature_points(),
                                    u_exact_solution);
    analytical_solution->pressure_value_list(fe_values.get_quadrature_points(),
                                             p_exact_solution);

    // Exact gradients: velocity and pressure
    std::vector<Tensor<2, dim>> u_exact_gradients(
            fe_values.n_quadrature_points,
            Tensor<2, dim>());
    std::vector<Tensor<1, dim>> p_exact_gradients(
            fe_values.n_quadrature_points,
            Tensor<1, dim>());
    analytical_solution->gradient_list(fe_values.get_quadrature_points(),
                                       u_exact_gradients);
    analytical_solution->pressure_gradient_list(
            fe_values.get_quadrature_points(), p_exact_gradients);

    for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
        // Integrate the square difference between exact and numeric solution
        // for function values and gradients (both pressure and velocity).
        Tensor<1, dim> diff_u = u_exact_solution[q] - u_solution_values[q];
        double diff_p = p_exact_solution[q] - p_solution_values[q];

        Tensor<2, dim> diff_u_gradient =
                u_exact_gradients[q] - u_solution_gradients[q];
        Tensor<1, dim> diff_p_gradient =
                p_exact_gradients[q] - p_solution_gradients[q];

        l2_error_integral_u += diff_u * diff_u * fe_values.JxW(q);
        l2_error_integral_p += diff_p * diff_p * fe_values.JxW(q);

        h1_error_integral_u +=
                scalar_product(diff_u_gradient, diff_u_gradient) *
                fe_values.JxW(q);
        h1_error_integral_p +=
                diff_p_gradient * diff_p_gradient * fe_values.JxW(q);
    }
}


template<int dim>
void ErrorStokesCylinder<dim>::
write_header_to_file(std::ofstream &file) {
    // u_H1 is the H1 norm of the error for the velocity, while
    // u_h1 is the H1 semi-norm of the error for the velocity.
    file << "h, \\|u\\|_{L^2}, \\|u\\|_{H^1}, |u|_{H^1}, "
            "\\|p\\|_{L^2}, \\|p\\|_{H^1}, |p|_{H^1}" << std::endl;
}


template<int dim>
void ErrorStokesCylinder<dim>::
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
class ErrorStokesCylinder<2>;

template
class ErrorStokesCylinder<3>;
