#include <deal.II/base/point.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "ErrorStokesCylinder.h"


template<int dim>
ErrorStokesCylinder<dim>::ErrorStokesCylinder(const double radius,
                                              const double half_length,
                                              const unsigned int n_refines,
                                              const int element_order,
                                              const bool write_output)
        : StokesCylinder<dim>(radius, half_length, n_refines, element_order,
                              write_output) {
    // Set the sphere outside the channel
    if (dim == 2) {
        this->center = Point<2>(2 * half_length, 0);
    } else {
        this->center = Point<dim>(3 * half_length, 0, 0);
    }
}


template<int dim>
double ErrorStokesCylinder<dim>::
compute_error() {
    this->run();
    return 0.1234;
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

    ErrorStokesCylinder<2> s3(radius, half_length, n_refines, elementOrder,
                              write_vtk);

    double error = s3.compute_error();
    std::cout << "Error was: " << error << std::endl;
}
