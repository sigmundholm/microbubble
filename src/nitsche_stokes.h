#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>


using namespace dealii;


namespace Stokes {
    template<int dim>
    class StokesNitsche {
    public:
        StokesNitsche(const unsigned int degree);

        virtual void run();

    protected:
        virtual void make_grid();

        void setup_dofs();

        void assemble_system();

        void solve();

        void output_results() const;

        const unsigned int degree;
        Triangulation<dim> triangulation;
        FESystem<dim> fe;
        DoFHandler<dim> dof_handler;

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        Vector<double> solution;
        Vector<double> system_rhs;
        double left_boundary;
        double radius;
    };
}
