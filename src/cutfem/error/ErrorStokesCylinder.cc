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
compute_error2() {
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

    double l2_error_integral = 0;
    double h1_error_integral = 0;

    for (const auto &cell : this->dof_handler.active_cell_iterators()) {
        cut_fe_values.reinit(cell);

        const boost::optional<const FEValues<dim> &> fe_values_inside =
                cut_fe_values.get_inside_fe_values();

        if (fe_values_inside) {
            integrate_cell(*fe_values_inside, l2_error_integral,
                           h1_error_integral);
        }
    }

    Error error;
    error.mesh_size = this->h;
    error.l2_error = pow(l2_error_integral, 0.5);
    error.h1_error = pow(h1_error_integral, 0.5);
    return error;
}


template<int dim>
void ErrorStokesCylinder<dim>::
integrate_cell(const FEValues<dim> &fe_values,
               double &l2_error_integral,
               double &h1_error_integral) const {

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    std::vector<Tensor<1, dim>> solution_values(fe_values.n_quadrature_points);
    std::vector<Tensor<2, dim>> gradients(fe_values.n_quadrature_points);

    fe_values[velocities].get_function_values(this->solution, solution_values);
    fe_values[velocities].get_function_gradients(this->solution, gradients);

    // Exact solution
    std::vector<Tensor<1, dim>> exact_solution(fe_values.n_quadrature_points,
                                               Tensor<1, dim>());
    analytical_solution->value_list(fe_values.get_quadrature_points(),
                                    exact_solution);

    // TODO calculate the gradient in the analytical solution too, for H1 norm.
    for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
        Tensor<1, dim> diff = exact_solution[q] - solution_values[q];
        // TODO test denne ved å kun integrere scalarfeltet f(x) = 1.
        l2_error_integral += diff * diff * fe_values.JxW(q);
    }
}


template<int dim>
void ErrorStokesCylinder<dim>::
write_header_to_file(std::ofstream &file) {
    file << "mesh_size, u_L2" << std::endl;
}


template<int dim>
void ErrorStokesCylinder<dim>::
write_error_to_file(Error &error, std::ofstream &file) {
    file << error.mesh_size << "," << error.l2_error << std::endl;
}


template
class ErrorStokesCylinder<2>;

template
class ErrorStokesCylinder<3>;
