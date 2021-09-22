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

#include "rhs.h"


using namespace dealii;
using namespace cutfem;



namespace examples::cut::HeatEquation {

    using NonMatching::LocationToLevelSet;


    template<int dim>
    class HeatEqn : public ScalarProblem<dim> {
    public:
        HeatEqn(const double nu,
                const double tau,
                const double radius,
                const double half_length,
                const unsigned int n_refines,
                const int element_order,
                const bool write_output,
                Function<dim> &rhs,
                Function<dim> &bdd_values,
                Function<dim> &analytical_soln,
                Function<dim> &levelset_func,
                const bool stabilized = true,
                const bool crank_nicholson = false);

        virtual Error
        run(unsigned int bdf_type, unsigned int steps,
            Vector<double> &supplied_solution);

        virtual Error
        run(unsigned int bdf_type, unsigned int steps);

        static void
        write_header_to_file(std::ofstream &file);

        static void
        write_error_to_file(Error &error, std::ofstream &file);

    protected:
        void
        set_bdf_coefficients(unsigned int bdf_type);

        void
        interpolate_first_steps(unsigned int bdf_type,
                                std::vector<Error> &errors);

        void
        make_grid(Triangulation<dim> &tria) override;

        virtual void
        assemble_local_over_cell(const FEValues<dim> &fe_values,
                                 const std::vector<types::global_dof_index> &loc2glb) override;

        virtual void
        assemble_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glb) override;

        void
        assemble_matrix();

        void
        assemble_matrix_local_over_cell(const FEValues<dim> &fe_values,
                                        const std::vector<types::global_dof_index> &loc2glb);

        void
        assemble_matrix_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glb);

        void
        assemble_rhs(int time_step);

        void
        assemble_rhs_local_over_cell(const FEValues<dim> &fe_values,
                                     const std::vector<types::global_dof_index> &loc2glb);

        void
        assemble_rhs_local_over_cell_cn(const FEValues<dim> &fe_values,
                                        const std::vector<types::global_dof_index> &loc2glb,
                                        const int time_step);

        void
        assemble_rhs_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glob);

        void
        assemble_rhs_local_over_surface_cn(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glob,
                const int time_step);

        Error
        compute_time_error(std::vector<Error> errors);

        void
        compute_condition_number();

        void
        write_time_header_to_file(std::ofstream &file);

        void
        write_time_error_to_file(Error &error, std::ofstream &file);

        const double nu;
        const double tau;

        const double radius;
        const double half_length;

        bool triangulation_exists = false;

        double condition_number = 0;

        std::vector<Vector<double>> solutions;
        std::vector<double> bdf_coeffs;
        const bool crank_nicholson;

    };

} // namespace examples::cut::HeatEquation

#endif //MICROBUBBLE_CUTFEM_POISSON_POISSON_H
