#ifndef MICROBUBBLE_FLOW_PROBLEM_H
#define MICROBUBBLE_FLOW_PROBLEM_H

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

#include "cutfem/errors/error_calculator.h"


using namespace dealii;
using namespace cutfem;

namespace utils::problems::flow {

    using NonMatching::LocationToLevelSet;


    struct Error {
        double h = 0;
        double tau = 0;
        double time_step = 0;
        double l2_error_u = 0;
        double h1_error_u = 0;
        double h1_semi_u = 0;
        double l2_error_p = 0;
        double h1_error_p = 0;
        double h1_semi_p = 0;
    };


    template<int dim>
    class FlowProblem {
    public:
        FlowProblem(const unsigned int n_refines,
                    const int element_order,
                    const bool write_output,
                    Function<dim> &levelset_func,
                    TensorFunction<1, dim> &analytic_v,
                    Function<dim> &analytic_p);

        virtual Error
        run_step();

        static void
        write_header_to_file(std::ofstream &file);

        static void
        write_error_to_file(Error &error, std::ofstream &file);

    protected:
        virtual void
        make_grid(Triangulation<dim> &tria) = 0;

        void
        setup_level_set();

        void
        setup_quadrature();

        void
        distribute_dofs();

        void
        initialize_matrices();

        virtual void
        assemble_system() = 0;

        virtual void
        assemble_local_over_bulk(const FEValues<dim> &fe_values,
                                 const std::vector<types::global_dof_index> &loc2glb) = 0;

        virtual void
        assemble_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glb) = 0;

        void
        solve();

        void
        output_results(bool minimal_output = false) const;

        Error
        compute_error();

        void
        integrate_cell(const FEValues<dim> &fe_v,
                       double &l2_error_integral_u,
                       double &h1_error_integral_u,
                       double &l2_error_integral_p,
                       double &h1_error_integral_p,
                       const double &mean_numerical_pressure,
                       const double &mean_exact_pressure) const;

        const unsigned int n_refines;
        bool write_output;

        Function<dim> *levelset_function;
        TensorFunction<1, dim> *rhs_function;
        TensorFunction<1, dim> *boundary_values;
        TensorFunction<1, dim> *analytical_velocity;
        Function<dim> *analytical_pressure;

        // Cell side-length.
        double h;
        double tau;
        const unsigned int element_order;

        Triangulation<dim> triangulation;
        FESystem<dim> mixed_fe;

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
        Vector<double> rhs;

        Vector<double> solution;

        AffineConstraints<double> constraints;
    };

} // namespace utils::problems::flow


#endif // MICROBUBBLE_FLOW_PROBLEM_H
