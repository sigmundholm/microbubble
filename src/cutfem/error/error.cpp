#include <iostream>

#include "ErrorStokesCylinder.h"


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

    double sphere_radius = radius / 4;
    double sphere_x_coord = half_length / 2;

    const int dim = 2;

    ErrorStokesRhs<dim> stokesRhs(radius, 2 * half_length, pressure_drop);
    ErrorBoundaryValues<dim> boundaryValues(radius, 2 * half_length, pressure_drop);

    ErrorStokesCylinder<dim> s3(radius, half_length, n_refines, elementOrder,
                                write_vtk, stokesRhs, boundaryValues,
                                pressure_drop, sphere_radius, sphere_x_coord);

    Error error = s3.compute_error();
    std::cout << "Error was: " << error.l2_error << std::endl;
}
