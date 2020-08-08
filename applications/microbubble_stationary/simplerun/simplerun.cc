#include <deal.II/base/point.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "../StokesCylinder.h"

int
main()
{
  const unsigned int n_subdivisions = 15;
  const unsigned int n_refines      = 0;
  const unsigned int elementOrder   = 1;

  printf("numRefines=%d\n", n_subdivisions);
  printf("elementOrder=%d\n", elementOrder);
  const bool write_vtk = true;

  StokesCylinder<2> s3(n_subdivisions, n_refines, elementOrder, write_vtk);
  s3.run();
}
