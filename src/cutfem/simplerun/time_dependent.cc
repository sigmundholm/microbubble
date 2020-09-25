#include <deal.II/base/point.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "../StokesCylinder.h"

int
main() {
    const unsigned int n_refines = 5;
    const int element_order = 1;

    printf("num_refines=%d\n", n_refines);
    printf("element_order=%d\n", element_order);
    const bool write_vtk = true;

    double radius = 0.205;
    double half_length = 1.1;
    double sphere_radius = radius / 4;
    double sphere_x_coord = -half_length / 2;

    const int dim = 2;

    StokesRhs<dim> stokes_rhs;
    BoundaryValues<dim> boundary_values(radius, 2 * half_length);
    InitialValues<dim> initial_values;

    StokesCylinder<dim> s(radius, half_length, n_refines, element_order,
                          write_vtk, stokes_rhs, boundary_values, initial_values, sphere_radius,
                          sphere_x_coord);
    s.run_time_dependent(0.1, 10);
}
