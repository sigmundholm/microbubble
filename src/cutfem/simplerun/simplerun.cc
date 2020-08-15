#include <deal.II/base/point.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "../StokesCylinder.h"

int
main() {
    const unsigned int n_subdivisions = 15;
    const unsigned int n_refines = 6;
    const unsigned int elementOrder = 1;

    printf("numRefines=%d\n", n_subdivisions);
    printf("elementOrder=%d\n", elementOrder);
    const bool write_vtk = true;

    double radius = 0.205;
    double half_length = 1.1;

    const int dim = 2;

    StokesRhs<dim> stokesRhs;
    BoundaryValues<dim> boundaryValues(radius, 2 * half_length);

    StokesCylinder<dim> s(radius, half_length, n_refines, elementOrder,
                         write_vtk, stokesRhs, boundaryValues);
    s.run();
}
