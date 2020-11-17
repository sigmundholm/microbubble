#ifndef MICROBUBBLE_INTEGRATION_H
#define MICROBUBBLE_INTEGRATION_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/function.h>

using namespace dealii;

// TODO make this into a library instead
namespace Utils {

    /**
     * Compute the mean of the numerical and the exact pressure over the domain.
     */
    template<int dim>
    void compute_mean_pressure(DoFHandler<dim> &dof,
                               NonMatching::FEValues<dim> &cut_fe_v,
                               Vector<double> &solution,
                               Function<dim> &pressure,
                               double &mean_numerical_pressure,
                               double &mean_analytical_pressure) {
        Assert(solution.size() == dof.n_dofs(),
               ExcDimensionMismatch(solution.size(), dof.n_dofs()));

        double area = 0;
        for (const auto &cell : dof.active_cell_iterators()) {
            cut_fe_v.reinit(cell);

            const boost::optional<const FEValues<dim> &> fe_v =
                    cut_fe_v.get_inside_fe_values();
            if (fe_v) {
                const FEValuesExtractors::Scalar p(dim);

                // Extract the numerical pressure values from the solution vector.
                std::vector<double> numerical(fe_v->n_quadrature_points);
                (*fe_v)[p].get_function_values(solution, numerical);

                // Extract the exact pressure values
                std::vector<double> exact(fe_v->n_quadrature_points);
                pressure.value_list(fe_v->get_quadrature_points(), exact);

                for (unsigned int q = 0; q < fe_v->n_quadrature_points; ++q) {
                    mean_numerical_pressure += numerical[q] * fe_v->JxW(q);
                    mean_analytical_pressure += exact[q] * fe_v->JxW(q);
                    area += fe_v->JxW(q);
                }
            }
        }
        mean_numerical_pressure /= area;
        mean_analytical_pressure /= area;
    }


} // namespace Utils


#endif //MICROBUBBLE_INTEGRATION_H
