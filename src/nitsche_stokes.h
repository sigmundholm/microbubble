#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>


using namespace dealii;


namespace Stokes {

    template<int dim>
    class RightHandSide : public TensorFunction<1, dim> {
    public:
        virtual double point_value(const Point<dim> &p, const unsigned int component = 0) const;

        void vector_value(const Point<dim> &p, Tensor<1, dim> &value) const;

        void value_list(const std::vector<Point<dim>> &points,
                        std::vector<Tensor<1, dim>> &values) const override;
    };


    template<int dim>
    class BoundaryValues : public TensorFunction<1, dim> {
    public:
        virtual double point_value(const Point<dim> &p, const unsigned int component) const;

        void vector_value(const Point<dim> &p, Tensor<1, dim> &value) const;

        void value_list(const std::vector<Point<dim>> &points,
                        std::vector<Tensor<1, dim>> &values) const override;
    };


    template<int dim>
    class StokesNitsche {
    public:
        StokesNitsche(const unsigned int degree, RightHandSide<dim> &rhs, BoundaryValues<dim> &bdd_val,
                      unsigned int do_nothing_bdd_id=1);

        virtual void run();

    protected:
        virtual void make_grid();

        void output_grid();

        void setup_dofs();

        void assemble_system();

        void solve();

        void output_results() const;

        const unsigned int degree;
        Triangulation<dim> triangulation;
        FESystem<dim> fe;
        DoFHandler<dim> dof_handler;
        RightHandSide<dim> *right_hand_side;
        BoundaryValues<dim> *boundary_values;
        const unsigned int do_nothing_bdd_id;

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        Vector<double> solution;
        Vector<double> system_rhs;
    };
}
