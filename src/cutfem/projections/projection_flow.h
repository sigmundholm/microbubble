#ifndef MICROBUBBLE_PROJECTIONS_PROJECTION_FLOW_H
#define MICROBUBBLE_PROJECTIONS_PROJECTION_FLOW_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
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

#include "../utils/flow_problem.h"

#include "rhs.h"


using namespace dealii;
using namespace cutfem;

namespace examples::cut::projections {

    using NonMatching::LocationToLevelSet;
    using namespace utils::problems::flow;

    template<int dim>
    class ProjectionFlow : public FlowProblem<dim> {
    public:
        ProjectionFlow(const double radius,
                       const double half_length,
                       const unsigned int n_refines,
                       const int element_order,
                       const bool write_output,
                       Function<dim> &levelset_func,
                       TensorFunction<1, dim> &analytic_vel,
                       Function<dim> &analytic_pressure,
                       const double sphere_radius, // TODO remove these
                       const double sphere_x_coord);

    protected:
        void
        make_grid(Triangulation<dim> &tria) override;

        void
        assemble_system() override;

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

} // namespace examples::cut::projections


#endif // MICROBUBBLE_PROJECTIONS_PROJECTION_FLOW_H
