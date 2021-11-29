#ifndef MICROBUBBLE_CUTFEM_POISSON_POISSON_H
#define MICROBUBBLE_CUTFEM_POISSON_POISSON_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_base.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <vector>

#include "cutfem/errors/error_calculator.h"

#include "rhs.h"
#include "../utils/scalar_problem.h"


using namespace dealii;
using namespace cutfem;


using NonMatching::LocationToLevelSet;
using namespace utils::problems::scalar;

template<int dim>
class Poisson : public ScalarProblem<dim> {
public:
    Poisson(const double radius,
            const double half_length,
            const unsigned int n_refines,
            const int element_order,
            const bool write_output,
            Function<dim> &rhs,
            Function<dim> &bdd_values,
            Function<dim> &analytical_soln,
            Function<dim> &domain_func,
            const bool stabilized = true);

    static void
    write_header_to_file(std::ofstream &file);

    static void
    write_error_to_file(ErrorBase *error, std::ofstream &file);

protected:
    void
    make_grid(Triangulation<dim> &tria) override;

    void
    assemble_local_over_cell(const FEValues<dim> &fe_values,
                             const std::vector<types::global_dof_index> &loc2glb) override;

    void
    assemble_local_over_surface(
            const FEValuesBase<dim> &fe_values,
            const std::vector<types::global_dof_index> &loc2glb) override;

    const double radius;
    const double half_length;
};


#endif //MICROBUBBLE_CUTFEM_POISSON_POISSON_H
