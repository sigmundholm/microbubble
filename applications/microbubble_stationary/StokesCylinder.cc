#include "StokesCylinder.h"

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/non_matching/cut_mesh_classifier.h>
#include <deal.II/non_matching/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/optional.hpp>

#include <cmath>
#include <fstream>

#include "cutfem/geometry/SignedDistanceSphere.h"
#include "cutfem/nla/sparsity_pattern.h"
#include "cutfem/stabilization/jump_stabilization.h"

using namespace cutfem;

unsigned int
compute_gammaD(const unsigned int element_order)
{
  return 10.0 / 2 * (element_order + 1) * element_order;
}

template <int dim>
StokesCylinder<dim>::StokesCylinder(const unsigned int n_subdivisions,
                                      const unsigned int n_refines,
                                      const int          element_order,
                                      const bool         write_output)
  : n_subdivisions(n_subdivisions)
  , n_refines(n_refines)
  , gammaA(.5)
  , gammaD(compute_gammaD(element_order))
  , write_output(write_output)
  , rhs_function(frequency_analytic_solution, center)
  , element_order(element_order)
  , fe_levelset(element_order)
  , levelset_dof_handler(triangulation)
  , dof_handler(triangulation)
  , cut_mesh_classifier(triangulation, levelset_dof_handler, levelset)
{
  h = 0;
  // Use no constraints when projecting.
  constraints.close();
}

template <int dim>
void
StokesCylinder<dim>::setup_quadrature()
{
  const unsigned int quadOrder = 2 * element_order + 1;
  q_collection.push_back(QGauss<dim>(quadOrder));
  q_collection1D.push_back(QGauss<1>(quadOrder));
}

template <int dim>
void
StokesCylinder<dim>::run()
{
  make_grid();
  setup_quadrature();
  setup_level_set();
  cut_mesh_classifier.reclassify();
  distribute_dofs();
  initialize_matrices();
  assemble_system();
  solve();
  if (write_output)
    {
      output_results();
    }
}

template <int dim>
void
StokesCylinder<dim>::make_grid()
{
  std::cout << "Creating triangulation" << std::endl;

  GridGenerator::subdivided_hyper_cube(triangulation,
                                       n_subdivisions,
                                       -1.5,
                                       1.5);
  triangulation.refine_global(n_refines);

  mapping_collection.push_back(MappingCartesian<dim>());

  // Save the cell-size, we need it in the Nitsche term.
  typename Triangulation<dim>::active_cell_iterator cell =
    triangulation.begin_active();
  h = std::pow(cell->measure(), 1.0 / dim);
}

template <int dim>
void
StokesCylinder<dim>::setup_level_set()
{
  std::cout << "Setting up level set" << std::endl;

  // The level set function lives on the whole background mesh.
  levelset_dof_handler.distribute_dofs(fe_levelset);
  printf("leveset dofs: %d\n", levelset_dof_handler.n_dofs());
  levelset.reinit(levelset_dof_handler.n_dofs());

  // Project the geometry onto the mesh.
  cutfem::geometry::SignedDistanceSphere<dim> signed_distance_sphere;
  VectorTools::project(levelset_dof_handler,
                       constraints,
                       QGauss<dim>(2 * element_order + 1),
                       signed_distance_sphere,
                       levelset);
}

template <int dim>
void
StokesCylinder<dim>::distribute_dofs()
{
  std::cout << "Distributing dofs" << std::endl;

  // We want to types of elements on the mesh
  // Lagrange elements and elements that are constant zero..
  fe_collection.push_back(FE_Q<dim>(element_order));
  fe_collection.push_back(FE_Nothing<dim>());

  // Set inside finite elements to FE_Q and outside to FE_nothing
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (LocationToLevelSet::OUTSIDE ==
          cut_mesh_classifier.location_to_level_set(cell))
        {
          // 1 is FE_nothing
          cell->set_active_fe_index(1);
        }
      else
        {
          // 0 is FE_Q
          cell->set_active_fe_index(0);
        }
    }
  dof_handler.distribute_dofs(fe_collection);
}

template <int dim>
void
StokesCylinder<dim>::initialize_matrices()
{
  solution.reinit(dof_handler.n_dofs());
  rhs.reinit(dof_handler.n_dofs());

  cutfem::nla::make_sparsity_pattern_for_stabilized(dof_handler,
                                                    sparsity_pattern);
  stiffness_matrix.reinit(sparsity_pattern);
}

template <int dim>
void
StokesCylinder<dim>::assemble_system()
{
  std::cout << "Assembling" << std::endl;

  // The stabilization is quite tricky to compute so this
  // is a helper object to do it.
  stabilization::JumpStabilization<dim> stabilization(dof_handler,
                                                      mapping_collection,
                                                      cut_mesh_classifier,
                                                      constraints);

  stabilization.set_function_describing_faces_to_stabilize(
    stabilization::inside_stabilization);
  stabilization.set_weight_function(stabilization::taylor_weights);



  NonMatching::RegionUpdateFlags region_update_flags;
  region_update_flags.inside = update_values | update_JxW_values |
                               update_gradients | update_quadrature_points;
  region_update_flags.surface = update_values | update_JxW_values |
                                update_gradients | update_quadrature_points |
                                update_normal_vectors;


  NonMatching::FEValues<dim> cut_fe_values(mapping_collection,
                                           fe_collection,
                                           q_collection,
                                           q_collection1D,
                                           region_update_flags,
                                           cut_mesh_classifier,
                                           levelset_dof_handler,
                                           levelset);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      const unsigned int n_dofs = cell->get_fe().dofs_per_cell;
      std::vector<types::global_dof_index> loc2glb(n_dofs);
      cell->get_dof_indices(loc2glb);

      // This call will compute quadrature rules relevant for this cell
      // in the background.
      cut_fe_values.reinit(cell);

      // Retrieve an FEValues object with quadrature points
      // over the full cell.
      const boost::optional<const FEValues<dim> &> fe_values_bulk =
        cut_fe_values.get_inside_fe_values();

      if (fe_values_bulk)
        assemble_local_over_bulk(*fe_values_bulk, loc2glb);

      // Retrieve an FEValues object with quadrature points
      // on the immersed surface.
      const boost::optional<const FEImmersedSurfaceValues<dim> &>
        fe_values_surface = cut_fe_values.get_surface_fe_values();

      if (fe_values_surface)
        assemble_local_over_surface(*fe_values_surface, loc2glb);

      // Compute and add stabilization.
      stabilization.compute_stabilization(cell);
      stabilization.add_stabilization_to_matrix(gammaA / (h * h),
                                                stiffness_matrix);
    }
}

template <int dim>
void
StokesCylinder<dim>::assemble_local_over_bulk(
  const FEValues<dim> &                       fe_values,
  const std::vector<types::global_dof_index> &loc2glb)
{
  const unsigned int n_dofs = fe_values.get_fe().dofs_per_cell;
  Vector<double>     rhs_loc(n_dofs);
  FullMatrix<double> gradigradj_loc(n_dofs, n_dofs);
  for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
    {
      for (unsigned int i = 0; i < n_dofs; ++i)
        {
          for (unsigned int j = 0; j < n_dofs; ++j)
            {
              gradigradj_loc(i, j) +=
                (fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q)) *
                fe_values.JxW(q);
            }
          const Point<dim> &point = fe_values.quadrature_point(q);
          rhs_loc(i) += rhs_function.value(point, 0) *
                        fe_values.shape_value(i, q) * fe_values.JxW(q);
        }
    }

  // map local to global.
  stiffness_matrix.add(loc2glb, gradigradj_loc);
  rhs.add(loc2glb, rhs_loc);
}



template <int dim>
void
StokesCylinder<dim>::assemble_local_over_surface(
  const FEImmersedSurfaceValues<dim> &        fe_values,
  const std::vector<types::global_dof_index> &loc2glb)
{
  const unsigned int n_dofs = fe_values.get_fe().dofs_per_cell;
  FullMatrix<double> dirichlet_loc(n_dofs, n_dofs);

  for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
    {
      const Tensor<1, dim> &normal = fe_values.normal_vector(q);
      for (unsigned int i = 0; i < n_dofs; ++i)
        {
          for (unsigned int j = 0; j < n_dofs; ++j)
            {
              dirichlet_loc(i, j) +=
                fe_values.JxW(q) * (-normal * fe_values.shape_grad(i, q) *
                                    fe_values.shape_value(j, q) +
                                    -normal * fe_values.shape_grad(j, q) *
                                    fe_values.shape_value(i, q) +
                                    gammaD / h * fe_values.shape_value(i, q) *
                                    fe_values.shape_value(j, q));
            }
        }
    }
  stiffness_matrix.add(loc2glb, dirichlet_loc);
}



template <int dim>
void
StokesCylinder<dim>::solve()
{
  std::cout << "Solving system" << std::endl;

  const int     maxIt = 2e4;
  double        tol   = 1.0e-10;
  SolverControl solver_control(maxIt, tol);
  SolverCG<>    solver(solver_control);
  solver.solve(stiffness_matrix, solution, rhs, PreconditionIdentity());
}

template <int dim>
void
StokesCylinder<dim>::output_results() const
{
  DataOut<dim, hp::DoFHandler<dim>> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("solution.vtk");
  data_out.write_vtk(output);

  // Output levelset function.
  DataOut<dim, DoFHandler<dim>> data_out_levelset;
  data_out_levelset.attach_dof_handler(levelset_dof_handler);
  data_out_levelset.add_data_vector(levelset, "levelset");
  data_out_levelset.build_patches();
  std::ofstream output_ls("levelset.vtk");
  data_out_levelset.write_vtk(output_ls);
}

template <int dim>
errors::Errors
StokesCylinder<dim>::compute_errors() const
{
  const StokesAnalytical<dim>  analytic(frequency_analytic_solution,
                                         center,
                                         sphere_radius);
  errors::AnalyticFunction<dim> analytic_wrapper(analytic);
  cutfem::errors::ErrorCalculator<dim, Vector<double>> error_calculator(
    levelset_dof_handler, levelset, dof_handler, cut_mesh_classifier);

  errors::Errors computed_errors =
    error_calculator.calculate_errors(solution, analytic_wrapper);
  return computed_errors;
}

template class StokesCylinder<2>;
template class StokesCylinder<3>;
