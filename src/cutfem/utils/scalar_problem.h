#ifndef MICROBUBBLE_SCALAR_PROBLEM_H
#define MICROBUBBLE_SCALAR_PROBLEM_H

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

#include "cutfem/geometry/SignedDistanceSphere.h"
#include "cutfem/nla/sparsity_pattern.h"
#include "cutfem/stabilization/jump_stabilization.h"


using namespace dealii;
using namespace cutfem;



namespace utils::problems::scalar {

    using NonMatching::LocationToLevelSet;


    struct Error {
        double h = 0;
        double tau = 0;
        double time_step = 0;
        double l2_error = 0;
        double h1_error = 0;
        double h1_semi = 0;
        double l_inf_l2_error = 0;
        double l_inf_h1_error = 0;
        double cond_num = 0;
    };


    template<int dim>
    class ScalarProblem {
    public:
        ScalarProblem(const unsigned int n_refines,
                      const int element_order,
                      const bool write_output,
                      Function<dim> &levelset_func,
                      Function<dim> &analytical_soln,
                      const bool stabilized = true);

        virtual Error
        run_step();

        Vector<double>
        get_solution();

        static void
        write_header_to_file(std::ofstream &file);

        static void
        write_error_to_file(Error &error, std::ofstream &file);

    protected:
        virtual void
        make_grid(Triangulation<dim> &tria) = 0;

        void
        setup_quadrature();

        void
        setup_level_set();

        void
        distribute_dofs();

        void
        initialize_matrices();

        virtual void
        assemble_system();

        virtual void
        assemble_local_over_cell(const FEValues<dim> &fe_values,
                                 const std::vector<types::global_dof_index> &loc2glb) = 0;

        virtual void
        assemble_local_over_surface(
                const FEValuesBase<dim> &fe_values,
                const std::vector<types::global_dof_index> &loc2glb) = 0;

        void
        solve();

        void
        output_results(std::string &suffix, bool minimal_output = false) const;

        void
        output_results(bool minimal_output = false) const;

        Error
        compute_error();

        Error
        compute_time_error(std::vector<Error> errors);

        void
        integrate_cell(const FEValues<dim> &fe_v,
                       double &l2_error_integral,
                       double &h1_error_integral) const;

        const unsigned int n_refines;
        bool write_output;

        Function<dim> *rhs_function;
        Function<dim> *boundary_values;
        Function<dim> *analytical_solution;
        Function<dim> *levelset_function;

        // Cell side-length.
        double h;
        double tau;
        const unsigned int element_order;

        const bool stabilized;

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
        Vector<double> rhs;
        Vector<double> solution;

        AffineConstraints<double> constraints;
    };

} // namespace utils::problems::scalar


#endif //MICROBUBBLE_SCALAR_PROBLEM_H
