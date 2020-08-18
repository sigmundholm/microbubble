#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/numerics/vector_tools.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "ErrorStokesCylinder.h"


using namespace dealii;


template<int dim>
ErrorStokesCylinder<dim>::ErrorStokesCylinder(const double radius,
                                              const double half_length,
                                              const unsigned int n_refines,
                                              const int element_order,
                                              const bool write_output,
                                              StokesRhs<dim> &rhs,
                                              BoundaryValues<dim> &bdd_values,
                                              const double pressure_drop,  // TODO remove
                                              const double sphere_radius,
                                              const double sphere_x_coord)
        : StokesCylinder<dim>(radius, half_length, n_refines, element_order,
                              write_output, rhs, bdd_values, sphere_radius,
                              sphere_x_coord),
          pressure_drop(pressure_drop) {

    // Set to some unused boundary_id for Dirichlet only
    // TODO fiks riktige outflow betingelser
    this->do_nothing_id = 1000;
}


template<int dim>
Error ErrorStokesCylinder<dim>::
compute_error() {
    this->run();
    std::cout << "Compute error" << std::endl;

    const ComponentSelectFunction <dim> pressure_mask(dim, dim + 1);
    const ComponentSelectFunction <dim> velocity_mask(std::make_pair(0, dim),
                                                      dim + 1);
    double length = 2 * this->half_length;

    AnalyticalSolution<dim> analytical_solution(this->radius, length,
                                                pressure_drop,
                                                this->sphere_x_coord,
                                                this->sphere_radius);

    Vector<double> cellwise_errors(this->triangulation.n_active_cells());

    QTrapez<1> q_trapez;
    QIterated <dim> quadrature(q_trapez, this->element_order + 2);

    VectorTools::integrate_difference(this->dof_handler,
                                      this->solution,
                                      analytical_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &velocity_mask);
    const double u_l2_error = VectorTools::compute_global_error(
            this->triangulation,
            cellwise_errors,
            VectorTools::L2_norm);
    std::cout << "  Error u: ||e_p||_L2 = " << u_l2_error << std::endl;

    // Find the L2 error from the ZeroFunction: some integral of the solution.
    VectorTools::integrate_difference(this->dof_handler,
                                      this->solution,
                                      Functions::ZeroFunction<dim>(dim + 1),
                                      cellwise_errors,
                                      QGauss<dim>(this->element_order + 2),
                                      VectorTools::L2_norm,
                                      &velocity_mask);
    const double u_l2_int = VectorTools::compute_global_error(
            this->triangulation,
            cellwise_errors,
            VectorTools::L2_norm);
    std::cout << "  Integral = " << u_l2_int << std::endl;

    VectorTools::integrate_difference(this->dof_handler,
                                      this->solution,
                                      analytical_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &pressure_mask);
    const double p_l2_error = VectorTools::compute_global_error(
            this->triangulation,
            cellwise_errors,
            VectorTools::L2_norm);
    std::cout << "  Error pressure = " << p_l2_error << std::endl;

    Error error;
    error.mesh_size = this->h;
    error.l2_error = u_l2_error;
    return error;
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
