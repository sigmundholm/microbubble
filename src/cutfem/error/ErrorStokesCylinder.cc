#include <deal.II/base/point.h>
#include <deal.II/numerics/vector_tools.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "ErrorStokesCylinder.h"


template<int dim>
ErrorStokesCylinder<dim>::ErrorStokesCylinder(const double radius,
                                              const double half_length,
                                              const unsigned int n_refines,
                                              const int element_order,
                                              const bool write_output,
                                              StokesRhs<dim> &rhs,
                                              BoundaryValues<dim> &bdd_values,
                                              const double pressure_drop)
        : StokesCylinder<dim>(radius, half_length, n_refines, element_order,
                              write_output, rhs, bdd_values),
          pressure_drop(pressure_drop) {
    // Set the sphere outside, as a quick hack to clear the channel.
    if (dim == 2) {
        this->center = Point<dim>(2 * half_length, 0);
    } else if (dim == 3) {
        this->center = Point<dim>(2 * half_length, 0, 0);
    }
}


template<int dim>
double ErrorStokesCylinder<dim>::
compute_error() {
    this->run();
    std::cout << "Compute error" << std::endl;


    // const ComponentSelectFunction <dim> pressure_mask(dim, dim + 1);
    const ComponentSelectFunction <dim> velocity_mask(std::make_pair(0, dim),
                                                      dim + 1);
    double length = 2 * this->half_length;

    AnalyticalSolution<dim> analytical_solution(this->radius, length,
                                                pressure_drop);

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

    std::cout << "  Errors: ||e_p||_L2 = " << u_l2_error << std::endl;
    return u_l2_error;
}


int main() {

    const unsigned int n_subdivisions = 15;
    const unsigned int n_refines = 4;
    const unsigned int elementOrder = 1;

    printf("numRefines=%d\n", n_subdivisions);
    printf("elementOrder=%d\n", elementOrder);
    const bool write_vtk = true;

    double radius = 0.205;
    double half_length = 1.1;
    double pressure_drop = 10;

    const int dim = 2;

    ErrorStokesRhs<dim> stokesRhs(radius, 2 * half_length, pressure_drop);
    ErrorBoundaryValues<dim> boundaryValues(radius, 2 * half_length, pressure_drop);

    ErrorStokesCylinder<dim> s3(radius, half_length, n_refines, elementOrder,
                                write_vtk, stokesRhs, boundaryValues,
                                pressure_drop);

    double error = s3.compute_error();
    std::cout << "Error was: " << error << std::endl;
}


template
class ErrorStokesCylinder<2>;

template
class ErrorStokesCylinder<3>;
