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
                        const bool write_output);

    double compute_error();
};


#endif // MICROBUBBLE_ERRORSTOKESCYLINDER_H
