#ifndef MICROBUBBLE_ERRORSTOKESCYLINDER_H
#define MICROBUBBLE_ERRORSTOKESCYLINDER_H

#include "ErrorRhs.h"


/**
 * Calculate the analytical error: Hagen-Poiseuille
 */
template<int dim>
class ErrorStokesCylinder : public StokesCylinder<dim> {
public:

    ErrorStokesCylinder(const double radius,
                        const double half_length,
                        const unsigned int n_refines,
                        const int element_order,
                        const bool write_output,
                        StokesRhs<dim> &rhs_function,
                        BoundaryValues<dim> &boundary_values,
                        const double pressure_drop);

    Error compute_error();

    static void write_header_to_file(std::ofstream &file);

    static void write_error_to_file(Error &error, std::ofstream &file);

private:
    double pressure_drop;

    // TODO override run for å legge til mer i matrisa etter assembly, for å fikse
    //  termene som skal legges til ved do-nothing
};


#endif // MICROBUBBLE_ERRORSTOKESCYLINDER_H
