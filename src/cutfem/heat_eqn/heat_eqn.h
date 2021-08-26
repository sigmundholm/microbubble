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


using namespace dealii;
using namespace cutfem;

using NonMatching::LocationToLevelSet;


namespace examples::cut::HeatEquation {


    template<int dim>
    class HeatEqn {
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
                Function<dim> &domain_func,
                const bool stabilized = true);

        virtual Error
        run(bool compute_cond_number, std::string suffix = "");

        static void
        write_header_to_file(std::ofstream &file);

        static void
        write_error_to_file(Error &error, std::ofstream &file);

    protected:
        void
        make_grid();

        void
        setup_level_set();

        void
        setup_quadrature();

        void
        distribute_dofs();

        void
        initialize_matrices();

        void
        assemble_system();

        void
        assemble_local_over_bulk(const FEValues<dim> &fe_values,
                                 const std::vector<types::global_dof_index> &loc2glb);

        void
        assemble_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glb);

        void
        solve();

        void
        output_results(std::string &suffix) const;

        Error
        compute_error();

        void
        compute_condition_number();

        void
        integrate_cell(const FEValues<dim> &fe_v,
                       double &l2_error_integral,
                       double &h1_error_integral) const;

        const double nu;
        const double tau;

        const double radius;
        const double half_length;
        const unsigned int n_refines;

        bool write_output;
        const bool stabilized;

        Function<dim> *rhs_function;
        Function<dim> *boundary_values;
        Function<dim> *analytical_solution;
        Function<dim> *domain_function;

        // Cell side-length.
        double h = 0;
        const unsigned int element_order;

        Triangulation<dim> triangulation;
        FE_Q<dim> fe;

        hp::FECollection<dim> fe_collection;
        hp::MappingCollection<dim> mapping_collection;
        hp::QCollection<dim> q_collection;
        hp::QCollection<1> q_collection1D;

        // Object managing degrees of freedom for the level set function.
        FE_Q<dim> fe_levelset;
        DoFHandler<dim> levelset_dof_handler;
        Vector<double> levelset;

        // Object managing degrees of freedom for the cutfem method.
        hp::DoFHandler<dim> dof_handler;

        NonMatching::CutMeshClassifier<dim> cut_mesh_classifier;

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> stiffness_matrix;
        double condition_number = 0;

        Vector<double> rhs;
        Vector<double> solution;

        AffineConstraints<double> constraints;
    };

} // namespace examples::cut::HeatEquation

#endif //MICROBUBBLE_CUTFEM_POISSON_POISSON_H
