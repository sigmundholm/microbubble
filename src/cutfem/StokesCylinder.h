#ifndef MICROBUBBLE_STOKESCYLINDER_H
#define MICROBUBBLE_STOKECYLINDER_H

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

#include "StokesRhs.h"
#include "cutfem/errors/error_calculator.h"


using namespace dealii;
using namespace cutfem;

using NonMatching::LocationToLevelSet;

template<int dim>
class StokesCylinder {
public:
    StokesCylinder(const double radius,
                   const double half_length,
                   const unsigned int n_refines,
                   const int element_order,
                   const bool write_output,
                   StokesRhs<dim> &rhs,
                   BoundaryValues<dim> &bdd_values,
                   const double sphere_radius,
                   const double sphere_x_coord);

    // Constructor for the time dependent problem
    StokesCylinder(const double radius,
                   const double half_length,
                   const unsigned int n_refines,
                   const int element_order,
                   const bool write_output,
                   StokesRhs<dim> &rhs,
                   BoundaryValues<dim> &bdd_values,
                   InitialValues<dim> &init_values,
                   const double sphere_radius,
                   const double sphere_x_coord);

    virtual void
    run_stationary();

    void
    run_time_dependent(double end_time, unsigned int steps);

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
    output_results(int time_step = 0) const;

    const double radius;
    const double half_length;
    const unsigned int n_refines;
    double k = 1;
    double time = 0;

    const double gammaA;
    const double gammaD;

    bool write_output;
    bool time_dependent = false;

    double sphere_radius;
    double sphere_x_coord;
    Point<dim> center;

    StokesRhs<dim> *rhs_function;
    BoundaryValues<dim> *boundary_values;
    InitialValues<dim> *initial_values;

    // Cell side-length.
    double h;
    const unsigned int element_order;

    unsigned int do_nothing_id = 2;

    Triangulation<dim> triangulation;
    FESystem<dim> stokes_fe;

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
    Vector<double> old_solution;

    AffineConstraints<double> constraints;
};


#endif // MICROBUBBLE_STOKESCYLINDER_H
