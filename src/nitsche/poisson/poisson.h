#ifndef MICROBUBBLE_NITSCHE_POISSON_POISSON_H
#define MICROBUBBLE_NITSCHE_POISSON_POISSON_H

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include "rhs.h"

using namespace dealii;


template<int dim>
class PoissonNitsche {
public:
    PoissonNitsche(const unsigned int degree,
                   const unsigned int n_refines);

    Error run();

    static void write_header_to_file(std::ofstream &file);

    static void write_error_to_file(Error &error, std::ofstream &file);

private:
    void make_grid();

    void setup_system();

    void assemble_system();

    void solve();

    void output_results() const;

    Error compute_error();

    const unsigned int degree;
    const unsigned int n_refines;

    double h = 0;

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;

};


#endif // MICROBUBBLE_NITSCHE_POISSON_POISSON_H
